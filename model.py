import tensorflow as tf
import numpy as np
import sys
from prepare_non_windowed_data import getEncoderDecoderData
from BatchIterator import *

NUM_ENC_FEATURES = 5
NUM_DEC_FEATURES = 5
NUM_LABELS = 48
MAX_STEPS = 22400
BATCH_SIZE = 16
NUM_EPOCHS = 1000
LR = 0.01
THRESHOLD = 0.5
CLASS_WEIGHTS = [1, 1]

SEED = 12340# set graph-level seed to make the random sequences generated by all ops be repeatable across sessions
tf.set_random_seed(SEED)
np.random.seed(SEED)

def encoderDecoderModel(encoderInput, \
        encoderOutput, \
        decoderInput, \
        decoderOutput, \
        seqlen, \
        state_size, \
        decoder_filters, \
        num_dec_features, \
        num_labels):
    '''
    Performs forward pass of encoder-decoder model.

    Returns:
    Logits of predicted decoder labels.
    '''

    num_samples, num_enc_steps = tf.shape(encoderInput)[0], tf.shape(encoderInput)[1]
    num_dec_steps = tf.shape(decoderInput)[1]

    ## Encoder
    cell = tf.contrib.rnn.LSTMCell(state_size,state_is_tuple=True)
    #rnn_inputs = tf.concat([tf.expand_dims(encoderOutput, axis=-1), encoderInput], axis=2)
    #rnn_inputs = encoderInput
    encoderOutput = tf.concat([tf.zeros([num_samples,1]), encoderOutput[:,:-1]], axis=1)
    rnn_inputs = tf.concat([tf.expand_dims(encoderOutput, axis=-1), encoderInput], axis=2)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen, dtype=tf.float32)
    rnn_outputs_shape = tf.shape(rnn_outputs)
    rnn_outputs = tf.reshape(rnn_outputs, [-1, rnn_outputs_shape[-1]])
    # Linear layer over encoder outputs
    W_enc = tf.get_variable('Wenc', [state_size, num_labels], dtype=tf.float32)
    b_enc = tf.get_variable('Benc', [1, num_labels], dtype=tf.float32)

    encoder_out = tf.matmul(rnn_outputs, W_enc) + b_enc

    ## Decoder
    W_dec_hl = tf.get_variable('Wdechl', [num_dec_features, decoder_filters], dtype=tf.float32)
    b_dec_hl = tf.get_variable('Bdechl', [decoder_filters], dtype=tf.float32)
    W_dec_labels = tf.get_variable('Wdeclbl', [decoder_filters], dtype=tf.float32)
    b_dec_labels = tf.get_variable('bdeclbl', [1], dtype=tf.float32)

    decoderInput = tf.reshape(decoderInput, [-1, num_labels, num_dec_features])
    decoderOutput = tf.reshape(decoderOutput, [-1, num_labels])

    decoder_hl = tf.tensordot(decoderInput, W_dec_hl, axes=[[2],[0]]) + b_dec_hl
    decoder_hl = tf.reshape(decoder_hl, [-1, num_labels, decoder_filters])
    decoder_out = tf.tensordot(decoder_hl, W_dec_labels, axes=[[2],[0]]) + b_dec_labels

    logits = encoder_out + decoder_out

    seqlen_mask = tf.slice(tf.gather(lower_triangular_ones, seqlen-1), [0, 0], [num_samples, num_dec_steps])
    seqlen_mask = tf.expand_dims(seqlen_mask, axis=-1)
    predictions = tf.nn.sigmoid(logits)
    decoderOutputLoss = -(CLASS_WEIGHTS[1]*decoderOutput * tf.log(predictions+1e-9) \
            + CLASS_WEIGHTS[0]*(1.0-decoderOutput) * tf.log(1-predictions+1e-9))
    #decoderOutputLoss = -(decoderOutput * tf.log(predictions + 1e-9))
    decoderOutputLoss = tf.reshape(decoderOutputLoss, [num_samples, num_dec_steps, num_labels])
    decoderOutputLoss *= seqlen_mask
    #decoderOutputLoss = tf.divide(tf.reduce_sum(decoderOutputLoss), tf.reduce_sum(seqlen_mask))
    decoderOutputLoss = tf.reduce_sum(decoderOutputLoss)
    predictions = tf.reshape(predictions, shape=[num_samples, num_dec_steps, num_labels])
    predictions *= seqlen_mask

    return decoderOutputLoss, seqlen_mask, predictions


X_enc = tf.placeholder(shape=[None, None, NUM_ENC_FEATURES],
        dtype=tf.float32) #Features of history data (encoder input)
Y_enc = tf.placeholder(shape=[None, None], 
        dtype=tf.float32) #Labels of history data (encoder output) 
X_dec = tf.placeholder(shape=[None, None, NUM_LABELS, NUM_DEC_FEATURES],
        dtype=tf.float32) #Features of next 24 hours, sampled by 5 mins (decoder input) 
Y_dec = tf.placeholder(shape=[None, None, NUM_LABELS],
        dtype=tf.float32) #Labels of next 24 hours, sampled by 5 mins (decoder output)
seqlen = tf.placeholder(tf.int32, shape=[None]) #List of lengths of each sequence
lower_triangular_ones = tf.constant(np.tril(np.ones([MAX_STEPS,MAX_STEPS])),dtype=tf.float32)


def train(encoderInput, encoderOutput, decoderInput, decoderOutput):
    '''
    Args:
    All arguments similar to that of encoderDecoderModel, except here
    these are python variables and contain entire training data.

    What this function does?
    1. Define loss as cross entropy loss between Y_dec and 
    logits returned by encoderDecoderModel() function.
    2. Define an optimizer to minimize above-defined loss.
    3. Create batches of size BATCH_SIZE of each of the arguments.
    4. Feed each batch to the encoderDecoderModel() function in a loop.
    5. Run this for NUM_EPOCHS epochs.

    Returns: None
    '''
    decoderOutputLoss, seqlen_mask, predictions = \
            encoderDecoderModel(X_enc, Y_enc, X_dec, Y_dec, seqlen, state_size=32, decoder_filters=3, num_dec_features=NUM_DEC_FEATURES, num_labels=NUM_LABELS)

    correct = tf.cast(tf.equal(tf.cast((predictions >= THRESHOLD),'float32'), Y_dec), 'float32')
    correct_pos = tf.multiply(correct, Y_dec)
    correct_pos = tf.multiply(correct_pos, seqlen_mask)
    incorrect_pos = tf.multiply(1.0 - correct, Y_dec)
    incorrect_pos = tf.multiply(incorrect_pos, seqlen_mask)
    total_pos = Y_dec
    total_neg = tf.multiply((1.0-Y_dec), seqlen_mask)
    correct_neg = tf.multiply(correct, (1.0-Y_dec))
    correct_neg = tf.multiply(correct_neg, seqlen_mask)
    incorrect_neg = tf.multiply(1.0-correct, 1.0-Y_dec)
    incorrect_neg = tf.multiply(incorrect_neg, seqlen_mask)
    accuracy_pos = tf.divide(tf.reduce_sum(tf.cast(correct_pos, 'float32')),
            tf.cast(tf.reduce_sum(total_pos), 'float32'))
    accuracy_neg = tf.divide(tf.reduce_sum(tf.cast(correct_neg, 'float32')),
            tf.cast(tf.reduce_sum(total_neg), 'float32'))

    train_variables = tf.trainable_variables()
    print(map(lambda x: x.op.name, train_variables))

    train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(decoderOutputLoss, var_list=train_variables)

    # run

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    sess.run(tf.global_variables_initializer())

    ## Train
    encoderInputItr = EncoderInputIterator(encoderInput)
    encoderOutputItr = EncoderOutputIterator(encoderOutput)
    decoderInputItr = DecoderInputIterator(decoderInput)
    decoderOutputItr = DecoderOutputIterator(decoderOutput)
    num_batches = int(encoderInputItr.size / BATCH_SIZE + (encoderInputItr.size%BATCH_SIZE>0))
    for epoch_no in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch_no in range(num_batches):
            encoderInputBch, encoderInputBchLens = encoderInputItr.batch(BATCH_SIZE)
            encoderOutputBch, encoderOutputBchLens = encoderOutputItr.batch(BATCH_SIZE)
            decoderInputBch, decoderInputBchLens = decoderInputItr.batch(BATCH_SIZE)
            decoderOutputBch, decoderOutputBchLens = decoderOutputItr.batch(BATCH_SIZE)
            #print('Epoch: {}, Batch: {},'.format(epoch_no, batch_no))
            decoderOutputLossRet, _= sess.run([decoderOutputLoss,train_op],
                    feed_dict={X_enc:encoderInputBch,
                        Y_enc:encoderOutputBch,
                        X_dec:decoderInputBch,
                        Y_dec:decoderOutputBch,
                        seqlen:encoderInputBchLens})
            epoch_loss += decoderOutputLossRet
            #print('Epoch:', epoch_no, 'Batch:', batch_no, 'loss:' , decoderOutputLossRet)

        if epoch_no%2==0:
            print('Epoch', epoch_no, 'completed, loss:', epoch_loss)
            encoderInputEvalItr = EncoderInputIterator(encoderInput)
            encoderOutputEvalItr = EncoderOutputIterator(encoderOutput)
            decoderInputEvalItr = DecoderInputIterator(decoderInput)
            decoderOutputEvalItr = DecoderOutputIterator(decoderOutput)
            correct_ones, correct_zeros, incorrect_ones, incorrect_zeros, total_ones, total_zeros = 0, 0, 0, 0, 0 ,0
            for batch_no in range(num_batches):
                encoderInputBch, encoderInputBchLens = encoderInputEvalItr.batch(BATCH_SIZE)
                encoderOutputBch, encoderOutputBchLens = encoderOutputEvalItr.batch(BATCH_SIZE)
                decoderInputBch, decoderInputBchLens = decoderInputEvalItr.batch(BATCH_SIZE)
                decoderOutputBch, decoderOutputBchLens = decoderOutputEvalItr.batch(BATCH_SIZE)
                correct_pos_ret, correct_neg_ret, incorrect_pos_ret, incorrect_neg_ret, total_pos_ret, total_neg_ret, \
                                =  sess.run([correct_pos, correct_neg, incorrect_pos, incorrect_neg, total_pos, total_neg],
                                        feed_dict={X_enc:encoderInputBch,
                                        Y_enc:encoderOutputBch,
                                        X_dec:decoderInputBch,
                                        Y_dec:decoderOutputBch,
                                        seqlen:encoderInputBchLens})
                correct_ones += np.sum(correct_pos_ret)
                incorrect_ones += np.sum(incorrect_pos_ret)
                correct_zeros += np.sum(correct_neg_ret)
                incorrect_zeros += np.sum(incorrect_neg_ret)
                total_ones += np.sum(total_pos_ret)
                total_zeros += np.sum(total_neg_ret)
            recall_1 = correct_ones*1.0/total_ones
            precision_1 = correct_ones*1.0/(correct_ones+incorrect_zeros)
            f1_score_1 = 2*recall_1*precision_1/(recall_1+precision_1)
            recall_0 = correct_zeros*1.0/total_zeros
            precision_0 = correct_zeros*1.0/(correct_zeros+incorrect_ones)
            f1_score_0 = 2*recall_0*precision_0/(recall_0+precision_0)
            print('Recall 1s:', recall_1)
            print('Precision 1s:', precision_1)
            print('Recall 0s:', recall_0)
            print('Precision 0s:', precision_0)
            print('F1 Score 1s:', f1_score_1)
            print('F1 Score 0s:', f1_score_0)
            print('--------------------')


def main():
    filePath = sys.argv[1]
    encoderInput, encoderOutput, decoderInput, decoderOutput \
            = getEncoderDecoderData(filePath)
    train(encoderInput, encoderOutput, decoderInput, decoderOutput)

if __name__ == '__main__':
    main()
