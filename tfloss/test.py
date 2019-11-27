import tensorflow as tf

def GenerateRandomPrediction (shape, deviation, data_seed, error_seed, errorf):
    labels = tf.random.uniform (shape=shape, min=0, max=deviation, seed=data_seed)
    logits = tf.add (labels, tf.random.uniform (shape=shape, min=0, max=deviation*errorf, seed=error_seed))

    return logits, labels

def TestHolomorphism (loss_function_1, loss_function_2):
    fail_count = 0
    
    for i in range (0, 1000, 10):
        logits, labels = GenerateRandomPrediction (shape=i, i, i, i, 0.0001)
        if loss_function_1 (logits, labels) != loss_function_2 (logits, labels):
            fail_count += 1
            print (f'[-] Test case {i+1} failed.')
        else:
            print (f'[+] Test case {i+1} passed.')
    
    print (f'{fail_count} cases failed.')

    return fail_count
    