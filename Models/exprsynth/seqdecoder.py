import numpy as np
from collections import Counter, namedtuple
from typing import Dict, Any, List, Tuple

import tensorflow as tf
from tensorflow.python.layers import core as tflayers_core
from dpu_utils.mlutils.vocabulary import Vocabulary
from dpu_utils.tfutils import pick_indices_from_probs

from exprsynth.model import Model, ModelTestResult, write_to_minibatch, collect_token_seq

START_TOKEN = '%start%'
END_TOKEN = '%end%'


SeqDecodingInformation = namedtuple("SeqDecodingInformation", ["rnn_state", "sequence", "seq_logprob"])


def __make_one_rnn_cell(cell_type: str, hidden_size: int, dropout_keep_rate):
    cell_type = cell_type.lower()
    if cell_type == 'lstm':
        cell = tf.contrib.rnn.LSTMCell(hidden_size)
    elif cell_type == 'gru':
        cell = tf.contrib.rnn.GRUCell(hidden_size)
    else:
        raise ValueError("Unknown RNN cell type '%s'!" % cell_type)

    cell = tf.contrib.rnn.DropoutWrapper(cell,
                                         output_keep_prob=dropout_keep_rate,
                                         state_keep_prob=dropout_keep_rate)
    return cell


def make_rnn_cell(layer_num: int, cell_type: str, hidden_size: int, dropout_keep_rate):
    cells = [__make_one_rnn_cell(cell_type, hidden_size, dropout_keep_rate)
             for _ in range(layer_num)]
    return tf.contrib.rnn.MultiRNNCell(cells)


class SeqDecoder(object):
    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = {
                    'decoder_vocab_size': 3000,
                    'decoder_max_target_length': 20,

                    'decoder_embedding_size': 64,

                    'decoder_rnn_cell_type': "GRU",
                    'decoder_rnn_hidden_size': 64,
                    'decoder_rnn_layer_num': 2,

                    'decoder_output_use_bias': False,
                   }
        return defaults

    def __init__(self, context_model: Model):
        # We simply share all internal data structures with the context model, so store it:
        self.__context_model = context_model
        pass

    @property
    def hyperparameters(self):
        return self.__context_model.hyperparameters

    @property
    def metadata(self):
        return self.__context_model.metadata

    @property
    def parameters(self):
        return self.__context_model.parameters

    @property
    def placeholders(self):
        return self.__context_model.placeholders

    @property
    def ops(self):
        return self.__context_model.ops

    @property
    def sess(self):
        return self.__context_model.sess

    def test_log(self, msg) -> None:
        return self.__context_model.test_log(msg)

    def make_parameters(self):
        decoder_token_vocab_size = self.hyperparameters['decoder_vocab_size']
        decoder_emb_size = self.hyperparameters['decoder_embedding_size']

        self.parameters['decoder_token_embedding'] = \
            tf.get_variable(name='decoder_token_embedding',
                            initializer=tf.glorot_uniform_initializer(),
                            shape=[decoder_token_vocab_size, decoder_emb_size])

        self.parameters['decoder_output_projection'] = \
            tflayers_core.Dense(decoder_token_vocab_size,
                                use_bias=self.hyperparameters['decoder_output_use_bias'],
                                name="decoder_output_projection",
                                )

    def make_placeholders(self, is_train: bool) -> None:
        if is_train:
            # original code
            # tensorized correct answer
            self.placeholders['target_token_ids'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=(None, self.hyperparameters['decoder_max_target_length']),
                               name="target_token_ids")

            self.placeholders['target_token_ids_mask'] = \
                tf.placeholder(dtype=tf.float32,
                               shape=(None, self.hyperparameters['decoder_max_target_length']),
                               name="target_token_ids_mask")

            # additional code for var misuse
            self.placeholders['target_var_ids'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=(None,),
                               name="target_var_ids")

            self.placeholders['target_var_positions'] = \
                tf.placeholder(dtype=tf.float32,
                               shape=(None,),
                               name="target_var_positions")
        else:
            self.placeholders['rnn_hidden_state'] = \
                tf.placeholder(dtype=tf.float32,
                               # Shape needs extra first dimension because the number of layers is variable:
                               shape=[None, 1, self.hyperparameters['decoder_rnn_hidden_size']],
                               name='rnn_hidden_state'
                              )
            self.placeholders['rnn_input_tok_id'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[1],
                               name='rnn_input_tok_id'
                              )

    def __make_decoder_rnn_initial_state(self, input_encoding: tf.Tensor, rnn_cell: tf.nn.rnn_cell.MultiRNNCell):
        initial_decoder_state = []
        for cell in rnn_cell._cells:
            cell_zero_state = cell.zero_state(self.placeholders['batch_size'], dtype=np.float32)
            if isinstance(cell_zero_state, tf.contrib.rnn.LSTMStateTuple):
                initial_decoder_state.append(cell_zero_state._replace(h=input_encoding))
            else:
                initial_decoder_state.append(input_encoding)
        initial_decoder_state = tuple(initial_decoder_state)

        return initial_decoder_state

    def make_model(self, is_train: bool=True):
        if is_train:
            self.__make_train_model()
        else:
            self.__make_test_model()

    def __make_test_model(self):
        print( "====================== making test cases" )
        rnn_cell = make_rnn_cell(self.hyperparameters['decoder_rnn_layer_num'],
                                 self.hyperparameters['decoder_rnn_cell_type'],
                                 hidden_size=self.hyperparameters['decoder_rnn_hidden_size'],
                                 dropout_keep_rate=self.placeholders['dropout_keep_rate'],
                                 )
        cur_output_tok_embedded = tf.nn.embedding_lookup(params=self.parameters['decoder_token_embedding'],
                                                         ids=self.placeholders['rnn_input_tok_id'])
        self.ops[ 'cur_output_tok_embedded' ] = cur_output_tok_embedded
        self.ops[ 'rnn_hidden_state' ] = self.placeholders['rnn_hidden_state']
        rnn_cell_state = tuple([self.placeholders['rnn_hidden_state'][layer,:,:]
                                for layer in range(self.hyperparameters['decoder_rnn_layer_num'])])
        cur_output, next_decoder_state = rnn_cell(cur_output_tok_embedded, rnn_cell_state)
        self.ops['one_rnn_decoder_step_output'] = tf.nn.softmax(self.parameters['decoder_output_projection'](cur_output))
        self.ops['one_rnn_decoder_step_state'] = next_decoder_state

    def __make_train_model(self):
        # use RNN to decode 
        #rnn_cell = make_rnn_cell(self.hyperparameters['decoder_rnn_layer_num'],
        #                         self.hyperparameters['decoder_rnn_cell_type'],
        #                         hidden_size=self.hyperparameters['decoder_rnn_hidden_size'],
        #                         dropout_keep_rate=self.placeholders['dropout_keep_rate'],
        #                         )
        # decoder_initial_state is the contexts for holes
        # shape [ num_exp, 64 ]
        #initial_cell_state = self.__make_decoder_rnn_initial_state(self.ops['decoder_initial_state'], rnn_cell)

        # target: input is word embedding for var position:
        #         output is the chosen var
        # Reorg data from [batch, time, emb_dim] to [time, batch, emb_dim], and build corresponding tensor array:
        #target_tokens_by_time = tf.transpose(self.placeholders['target_token_ids'], perm=[1, 0])
        #target_tokens_ta = tf.TensorArray(dtype=tf.int32,
        #                                  size=self.hyperparameters['decoder_max_target_length'],
        #                                  name="target_tokens_embedded_ta",
        #                                  )
        #target_tokens_ta = target_tokens_ta.unstack(target_tokens_by_time)

        # First, initialise loop variables:
        # starting at a wrong place?? looks like it is fine if the deminsion matches
        one_one_per_sample = tf.ones_like(self.placeholders['target_token_ids'][:,0])
        # [1] * num_of_exp
        self.ops[ 'one_one_per_sample' ] = one_one_per_sample
        initial_input0 = one_one_per_sample * self.metadata['decoder_token_vocab'].get_id_or_unk(START_TOKEN)
        # for each exapmle, generate from start symbal
        initial_input = tf.nn.embedding_lookup(self.parameters['decoder_token_embedding'], initial_input0)
        end_token = one_one_per_sample * self.metadata['decoder_token_vocab'].get_id_or_unk(END_TOKEN)
        #empty_output_logits_ta = tf.TensorArray(dtype=tf.float32,
        #                                        size=self.hyperparameters['decoder_max_target_length'],
        #                                        name="output_logits_ta",
        #                                        )

        #def condition(time_unused, output_logits_ta_unused, decoder_state_unused, last_output_tok_embedded_unused, finished):
        #    # reduce all: per-element and for all elem in finished
        #    # logical_not: make it capable to control loop
        #    return tf.logical_not(tf.reduce_all(finished))

        #def body(step, output_logits_ta, decoder_state, last_output_tok_embedded, finished):
        #    next_step = step + 1

        #    # Use the RNN to decode one more tok:
        #    cur_output, next_decoder_state = rnn_cell(last_output_tok_embedded, decoder_state)
        #    cur_rnn_output_logits = self.parameters['decoder_output_projection'](cur_output)

        #    # Decide if we're done everywhere:
        #    next_finished = tf.logical_or(finished, next_step >= self.hyperparameters['decoder_max_target_length'])

        #    # Decide next token: If in training, use the next target token...
        #    all_next_finished = tf.reduce_all(next_finished)
        #    cur_output_tok = tf.cond(all_next_finished,
        #                             lambda: end_token,
        #                             lambda: target_tokens_ta.read(step))

        #    cur_output_tok_embedded = tf.nn.embedding_lookup(self.parameters['decoder_token_embedding'],
        #                                                     cur_output_tok)

        #    # Write out the collected wisdom:
        #    output_logits_ta = output_logits_ta.write(step, cur_rnn_output_logits)
        #    return (next_step, output_logits_ta, next_decoder_state, cur_output_tok_embedded, next_finished)

        # Uses 2 layer GLU/LSTM RNN node
        # both condition and body takes loop_vars as input
        # and body returns the same arity and type as loop_vars
        #(_, final_output_logits_ta, _, _, _) = \
        #    tf.while_loop(condition,
        #                  body,
        #                  loop_vars=[tf.constant(0, dtype=tf.int32),
        #                             empty_output_logits_ta,
        #                             initial_cell_state,
        #                             initial_input,
        #                             tf.zeros_like(self.placeholders['target_token_ids'][:,0], dtype=tf.bool),
        #                             ],
        #                  parallel_iterations=1
        #                  )
 
        self.ops['initial_input'] = initial_input
        self.ops['initial_input0'] = initial_input0
        # try skip the while loop :)
        # this would only train 1 step, which would for sure have a huge loss
        #(_, final_output_logits_ta, _, _, _) = body( tf.constant(0, dtype=tf.int32), empty_output_logits_ta, initial_cell_state, \
        #                                             initial_input, tf.zeros_like(self.placeholders['target_token_ids'][:,0], dtype=tf.bool) )

        # put var embedding as init input
        # output is if [exp count, vocab size] shape
        # TODO: we don't need vocab size to be 3000?
        final_output_logits_ta2 = \
            tf.layers.dense(self.ops['decoder_initial_state'], # input embeddings
                            units=self.hyperparameters['decoder_vocab_size'],
                            use_bias=False,
                            activation=None,
                            kernel_initializer=tf.glorot_uniform_initializer(),
                            name="varchoice_representation_to_logits",
                            )  # Shape: [VP, num_variable_choices, 1]
        #return tf.squeeze(varchoice_logits, axis=-1)  # Shape: [VP, num_variable_choices]
        self.ops[ 'dense_output' ] = final_output_logits_ta2

        #used for RNN
        #output_logits_by_time = final_output_logits_ta.stack()
        # reshape from [15, 3000] to [15, 1, 3000]
        output_logits_by_time2 = tf.expand_dims(final_output_logits_ta2,1)
        # self.ops[ 'output_logits_by_time' ] = output_logits_by_time
        self.ops[ 'output_logits_by_time2' ] = output_logits_by_time2

        # tranpose, and change dimentions to [15,20,3000] from [20, 3000, 15]
        # 3000 is the max vocab size distribution?
        # 20 is the target size
        # 15 looks like the number of class
        #self.ops['decoder_output_logits'] = tf.transpose(output_logits_by_time, perm=[1, 0, 2])
        self.ops['decoder_output_logits2'] = output_logits_by_time2
        self.ops['decoder_output_probs'] = tf.nn.softmax(self.ops['decoder_output_logits2'])
        #self.ops['decoder_output_probs'] = tf.nn.softmax(self.ops['decoder_output_logits'])
        # shape [15, 20], have the form of [start_token, ..., end_token]
        self.ops[ 'target' ] = self.placeholders['target_token_ids']
        self.ops[ 'target_mask' ] = self.placeholders['target_token_ids_mask']

        # Produce loss:
        # reshape target to [15,1], we only care the first generated val
        den_correct = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.slice(self.placeholders['target_token_ids'], [0,0], [self.placeholders['batch_size'],1]),
                                                                     logits=self.ops['decoder_output_logits2'])
        #outputs_correct_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.placeholders['target_token_ids'],
        #                                                                          logits=self.ops['decoder_output_logits'])
        #outputs_correct_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.constant([1]*15),
        #                                                                          logits=self.ops['decoder_output_logits'])
        #masked_outputs_correct_crossent = outputs_correct_crossent * self.placeholders['target_token_ids_mask']


        #decoder_loss = tf.reduce_sum(masked_outputs_correct_crossent)
        decoder_loss = tf.reduce_sum( den_correct )
        self.ops['log_probs'] = -decoder_loss

        # Normalize by batch size:
        self.ops['loss'] = decoder_loss / tf.to_float(self.placeholders['batch_size'])

    @staticmethod
    def init_metadata(raw_metadata: Dict[str, Any]) -> None:
        # original code
        raw_metadata['decoder_token_counter'] = Counter()

        # additional code to process var occurrence
        raw_metadata[ 'var_name_counter' ] = Counter()

    @staticmethod
    def load_metadata_from_sample(raw_sample: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        # original code
        symbol_id_to_label = raw_sample['SymbolLabels']

        for symbol_label in symbol_id_to_label.values():
            raw_metadata['decoder_token_counter'][symbol_label] += 1

        # addtional code to process var occurrence
        var_name_to_id = raw_sample[ 'VariableOccurence' ]
        for name in var_name_to_id:
            raw_metadata[ 'var_name_counter' ][ name ] += len( var_name_to_id[ name ] )

    def finalise_metadata(self, raw_metadata_list: List[Dict[str, Any]], final_metadata: Dict[str, Any]) -> None:
        # original code
        # First, merge all needed information:
        merged_token_counter = Counter()
        for raw_metadata in raw_metadata_list:
            merged_token_counter += raw_metadata['decoder_token_counter']

        final_metadata['decoder_token_vocab'] = \
            Vocabulary.create_vocabulary(merged_token_counter,
                                         max_size=self.hyperparameters['decoder_vocab_size'] - 2, count_threshold=1)
        final_metadata['decoder_token_vocab'].add_or_get_id(START_TOKEN)
        final_metadata['decoder_token_vocab'].add_or_get_id(END_TOKEN)

        # addtional code to process var occurrence
        merged_var_counter = Counter()
        for raw_metadata in raw_metadata_list:
            merged_var_counter += raw_metadata[ 'var_name_counter' ]
        final_metadata[ 'var_occurrence_vocab' ] = \
            Vocabulary.create_vocabulary( merged_var_counter,
                                          max_size=self.hyperparameters[ 'decoder_vocab_size'] )
        print( "TOTAL VARIABLE VOCAB SIZE: ", len( final_metadata[ 'var_occurrence_vocab' ].id_to_token ) )

    @staticmethod
    def load_data_from_sample(hyperparameters: Dict[str, Any], metadata: Dict[str, Any], raw_sample: Dict[str, Any], result_holder: Dict[str, Any], is_train: bool=True) -> None:
        # original code
        prod_root_node = min(int(v) for v in raw_sample['Productions'].keys())
        sample_token_seq = []
        collect_token_seq(raw_sample, prod_root_node, sample_token_seq)
        #import pprint
        #pprint.pprint( raw_sample['Productions'] )
        #pprint.pprint( raw_sample['SymbolLabels'] )
        #print( "sample_token_seq:", sample_token_seq )

        max_len = hyperparameters['decoder_max_target_length']
        end_token_id = metadata['decoder_token_vocab'].get_id_or_unk(END_TOKEN)
        token_seq_tensorised = [metadata['decoder_token_vocab'].get_id_or_unk(token)
                                for token in sample_token_seq[:max_len - 1]]
        token_seq_tensorised.append(end_token_id)
        # HACK: only use one prediction
        #token_seq_mask = [1] * len(token_seq_tensorised)
        print( "=============================" )
        print( "replace token mask to 1 " )
        print( "=============================" )
        #token_seq_mask = [1] * len(token_seq_tensorised)
        token_seq_mask = [1] 
        #padding_size = max_len - len(token_seq_tensorised)
        padding_size = max_len - 1
        # TODO: the first token is actually start token
        #token_seq_tensorised = token_seq_tensorised + [end_token_id] * padding_size
        token_seq_tensorised = token_seq_tensorised[:1] + [end_token_id] * padding_size
        assert all(0<=t<len(metadata['decoder_token_vocab']) for t in token_seq_tensorised)
        token_seq_mask = token_seq_mask + [0] * padding_size

        print( "=============target token tensorized:", token_seq_tensorised )
        print( "=============target token un-tensorized:", sample_token_seq[:max_len - 1] )
        result_holder['target_token_ids'] = np.array(token_seq_tensorised, dtype=np.int32)
        result_holder['target_token_ids_mask'] = np.array(token_seq_mask, dtype=np.int32)
        if not is_train:
            result_holder['target_tokens'] = sample_token_seq

        # additional code to handle var occurrence
        # Here for convenience, just put all the node as well as name in 2 sequence
        var_name_tensorized = [] # this is the answer, represented in vocab_id
        var_position_tensorized = [] # this is the token id, which should be used to fetch embedding
        var_occurence = raw_sample[ 'VariableOccurence' ]
        for var_name in var_occurence:
            vocab_id = metadata['var_occurrence_vocab'].get_id_or_unk( var_name ) 
            var_name_tensorized += [ vocab_id ] * len( var_occurence[ var_name ] )
            var_position_tensorized += var_occurence[ var_name ]

        result_holder[ 'target_var_ids' ] = np.array( var_name_tensorized, dtype=np.int32 )
        result_holder[ 'target_var_positions' ] = np.array( var_position_tensorized, dtype=np.int32 )
        
    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        # original code
        batch_data['target_token_ids'] = []
        batch_data['target_token_ids_mask'] = []

        # additional for var misuse
        batch_data[ 'target_var_ids' ] = np.array([])
        batch_data[ 'target_var_positions' ] = np.array([])

    # fetching this function from contextgraphmodel
    def _get_number_of_nodes_in_graph(self, sample: Dict[str, Any]) -> int:
        if self.hyperparameters['cg_node_label_embedding_style'].lower() == 'token':
            return len(sample['cg_node_label_token_ids'])
        elif self.hyperparameters['cg_node_label_embedding_style'].lower() == 'charcnn':
            return len(sample['cg_node_label_unique_indices'])
        else:
            raise Exception("Unknown node label embedding style '%s'!"
                            % self.hyperparameters['cg_node_label_embedding_style'])

    def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> None:
        # original code
        batch_data['target_token_ids'].append(sample['target_token_ids'])
        batch_data['target_token_ids_mask'].append(sample['target_token_ids_mask'])

        # additional for var misuse
        original_cg_node_offset = batch_data['cg_node_offset'] - self._get_number_of_nodes_in_graph(sample)
        batch_data[ 'target_var_ids' ] = np.concatenate( ( batch_data[ 'target_var_ids' ], sample[ 'target_var_ids' ] ) )
        batch_data[ 'target_var_positions' ] = np.concatenate( ( batch_data[ 'target_var_positions' ],
                                                                 sample[ 'target_var_positions' ] + original_cg_node_offset ) )

    def finalise_minibatch(self, batch_data: Dict[str, Any], minibatch: Dict[tf.Tensor, Any], is_train: bool) -> None:
        # original code
        write_to_minibatch(minibatch, self.placeholders['target_token_ids'], batch_data['target_token_ids'])
        write_to_minibatch(minibatch, self.placeholders['target_token_ids_mask'], batch_data['target_token_ids_mask'])

        # additional for var misuse
        write_to_minibatch( minibatch, self.placeholders['target_var_ids'], batch_data['target_var_ids'] )
        write_to_minibatch( minibatch, self.placeholders['target_var_positions'], batch_data['target_var_positions'] )
        #import pdb
        #pdb.set_trace()
        

    def generate_suggestions_for_one_sample(self,
                                            test_sample: Dict[str, Any],
                                            test_sample_encoded: tf.Tensor,
                                            beam_size: int=3,
                                            max_decoding_steps: int=100) -> ModelTestResult:

        def expand_sequence(decoder_info: SeqDecodingInformation) -> List[SeqDecodingInformation]:
            last_tok = decoder_info.sequence[-1]
            if last_tok == END_TOKEN:
                return [decoder_info]

            last_tok_id = self.metadata['decoder_token_vocab'].get_id_or_unk(last_tok)
            rnn_one_step_data_dict = {
                self.placeholders['rnn_hidden_state']: decoder_info.rnn_state,
                self.placeholders['rnn_input_tok_id']: [last_tok_id],
                self.placeholders['dropout_keep_rate']: 1.0,
            }

            (output_probs, next_state) = \
                self.sess.run([self.ops['one_rnn_decoder_step_output'], self.ops['one_rnn_decoder_step_state']],
                              feed_dict=rnn_one_step_data_dict)
            next_tok_indices = pick_indices_from_probs(output_probs[0, :], beam_size)

            result = []
            for next_tok_idx in next_tok_indices:
                next_tok = self.metadata['decoder_token_vocab'].id_to_token[next_tok_idx]
                next_tok_prob = output_probs[0,next_tok_idx]
                new_decoder_info = SeqDecodingInformation(rnn_state=next_state,
                                                          sequence=list(decoder_info.sequence) + [next_tok],
                                                          seq_logprob=decoder_info.seq_logprob + np.log(next_tok_prob))
                result.append(new_decoder_info)

            return result

        rnn_cell = make_rnn_cell(self.hyperparameters['decoder_rnn_layer_num'],
                                 self.hyperparameters['decoder_rnn_cell_type'],
                                 hidden_size=self.hyperparameters['decoder_rnn_hidden_size'],
                                 dropout_keep_rate=1,
                                 )
        initial_cell_state = self.__make_decoder_rnn_initial_state(test_sample_encoded, rnn_cell)
        initial_decoder_info = SeqDecodingInformation(rnn_state=initial_cell_state,
                                                      sequence=[START_TOKEN],
                                                      seq_logprob=0.0)
        beams = [initial_decoder_info]  # type: List[SeqDecodingInformation]
        number_of_steps = 0
        while number_of_steps < max_decoding_steps and any(b.sequence[-1] != END_TOKEN for b in beams):
            new_beams = [new_beam
                         for beam in beams
                         for new_beam in expand_sequence(beam)]
            beams = sorted(new_beams, key=lambda b: -b.seq_logprob)[:beam_size]  # Pick top K beams

        self.test_log("Groundtruth: %s" % (" ".join(test_sample['target_tokens']),))

        all_predictions = []  # type: List[Tuple[List[str], float]]
        for (k, beam_info) in enumerate(beams):
            beam_info.sequence.pop()  # Remove END_TOKEN
            beam_info.sequence.pop(0)  # Remove START_TOKEN
            kth_result = beam_info.sequence
            all_predictions.append((kth_result, np.exp(beam_info.seq_logprob)))
            self.test_log("  @%i Prob. %.3f: %s" % (k+1, np.exp(beam_info.seq_logprob), " ".join(kth_result)))

        if len(beams) == 0:
            print("No beams finished!")

        return ModelTestResult(test_sample['target_tokens'], all_predictions)
