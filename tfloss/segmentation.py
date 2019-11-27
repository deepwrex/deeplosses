import tensorflow as tf 

def WeightedCrossEntropyLossFunctionProducer (beta):
    """
    Creates a weighted cross entropy function (computation graph builder) 
    with the given beta value and returns it. The returned function accepts 
    only logits and labels.

    beta -> beta value to be used for WCE

    Returns a function 
    """
    
    def WeightedCrossEntropy (logits, labels):
        """Weighted cross entropy

        logits -> y pred
        labels -> y ground truth

        Returns loss as tensorflow.Tensor
        """
        
        assert logits.size == labels.size 

        left_term  = tf.multiply (tf.multiply (beta, labels), tf.math.log (logits))
        right_term = tf.multiply (1 - labels, tf.math.log (1 - logits))
        return tf.negative (tf.add (left_term, right_term))

    return WeightedCrossEntropy


def BalancedCrossEntropyLossFunctionProducer (beta):
    """
    Creates a balanced cross entropy function (computation graph builder) 
    with the given beta value and returns it. The returned function accepts 
    only logits and labels.

    beta -> beta value to be used for WCE

    Returns a function 
    """
    
    def BalancedCrossEntropy (logits, labels):
        """Balanced cross entropy
        
        Similar to WCE, just that negative examples are also weighted

        logits -> y pred
        labels -> y ground truth

        Returns loss as tensorflow.Tensor
        """

        assert logits.size == labels.size 
        
        left_term  = tf.multiply (tf.multiply (beta, labels), tf.math.log (logits))
        right_term = tf.multiply (1 - beta, tf.multiply (1 - labels, tf.math.log (1 - logits)))
        return tf.negative (tf.add (left_term, right_term))

    return BalancedCrossEntropy

def FocalLossFunctionProducer (alpha, gamma):
    """
    Creates a focal loss function (computation graph builder) with the 
    given beta value and returns it. The returned function accepts only 
    logits and labels.

    alpha -> alpha value to be used for focal loss
    gamma -> gamma value to be used for focal loss 

    Returns a function 
    """

    def FocalLoss (logits, labels):
        """Focal loss

        It tries to down weight the contribution of easy examples so that 
        the network focuses more on the hard examples

        logits -> y pred
        labels -> y ground truth

        Returns loss as tensorflow.Tensor
        """

        assert logits.size == labels.size 
        
        left_term  = tf.multiply (tf.multiply (alpha, tf.pow (1 - logits, gamma)), 
            tf.multiply (labels, tf.math.log (logits)))
        right_term = tf.multiply (tf.multiply (1 - alpha, tf.pow (logits, gamma)), 
            tf.multiply (1 - labels, tf.math.log (1 - logits)))
        return tf.negative (tf.add (left_term, right_term))
        
    return FocalLoss


def DiceLoss (logits, labels):
    """Dice loss or F1 score

    Dice loss is similar ro Jaccard Index (IoU)

    logits -> y pred
    labels -> y ground truth
    
    Returns loss as tf.Tensor
    """

    return 1 - tf.divide (tf.add (tf.multiply (2 * labels, logits), 1), tf.add (1 + logits, labels))

def TverskyLossFunctionProducer (beta):
    """
    Creates a Tversky loss function (computation graph builder) with the 
    given beta value and returns it. The returned function accepts only 
    logits and labels.

    beta -> beta value to be used for focal loss

    Returns a function 
    """


    def TverskyLoss (logits, labels):
        """Tversky loss 

        The Tversky Index is a generalization of Dice coefficient

        logits -> y pred
        labels -> y ground truth

        Returns loss as tf.Tensor
        """

        pp_  = tf.multiply (logits, labels)      # p * p'
        pp_1 = tf.multiply (1 - labels, logits)  # (1 - p) * p'
        pp_2 = tf.multiply (1 - logits, labels)  # (1 - p') * p
        return tf.divide (pp_, tf.add (pp_, tf.multiply (beta, pp_1), tf.multiply (1 - beta, pp_2)))

    return TverskyLoss