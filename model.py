
X_enc = tf.placeholder(shape=[BATCH_SIZE, ENCODER_LEN, NUM_ENC_FEATURES],
        dtype=tf.float32) #Features of history data (encoder input)
Y_enc = tf.placeholder(shape=[BATCH_SIZE, ENCODER_LEN, 1], 
        dtype=tf.int32) #Labels of history data (encoder output) 
X_dec = tf.placeholder(shape=[BATCH_SIZE, DECODER_LEN, NUM_DEC_FEATURES],
        dtype=tf.float32) #Features of next 24 hours, sampled by 5 mins (decoder input) 
Y_dec = tf.placeholder(shape=[BATCH_SIZE, DECODER_LEN, 1],
        dtype=tf.int32) #Labels of next 24 hours, sampled by 5 mins (decoder output)

def encoderDecoderModel():
    '''
    Performs forward pass of encoder-decoder model.

    Returns:
    Logits of predicted decoder labels.
    '''

    return logits

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
    logits = encoderDecoderModel()
