import theano
import theano.tensor as T

def meanAveragePrecisionTheano(targets, predictions):
    
    outputs, updates = theano.scan(fn=avgPrecisionTheano,
                                    sequences = [targets.transpose(), 
                                                 predictions.transpose()],
                                    outputs_info=[None],)
        
    minAveragePrecision = outputs.min()
    meanAveragePrecision = outputs.mean()
    maxAveragePrecision = outputs.max()
    return (minAveragePrecision, meanAveragePrecision, maxAveragePrecision, outputs[0],outputs[1],outputs[2],outputs[3],outputs[4],outputs[5],outputs[6],outputs[7],outputs[8],outputs[9],outputs[10],outputs[11]) 
def everyAveragePrecisionTheano(targets, predictions):
   
   outputs, updates = theano.scan(fn=avgPrecisionTheano,sequences = [targets.transpose(),predictions.transpose()], outputs_info=[None],)
                                  
   return outputs	

def avgPrecisionTheano(targets, predictions):
    
    # Sort targets and predictions by predictions
    indices = T.argsort(predictions)
    Y = targets.take(indices)
    Y_hat = predictions.take(indices)
    
    # Compute every recall-precision values pair
    totalNbPos = T.eq(Y,1).sum().astype("floatX")
    
    def innerFunction(seq, out1, out2, out3, nonSeq, nonSeq2):
        index = seq
        areaUnderCurve = out1
        previousRecall = out2
        maxPrecision = out3
        Y = nonSeq
        Y_hat = nonSeq2
                
        nbTP = ((Y_hat >= Y_hat[index]) * T.eq(Y,1)).sum().astype("floatX")
        nbFP = ((Y_hat >= Y_hat[index]) * T.eq(Y,0)).sum().astype("floatX")
        
        recall = nbTP / totalNbPos
        precision = T.switch(T.eq(nbTP,0), 0, nbTP / (nbTP + nbFP))
        
        newAreaUnderCurve = areaUnderCurve + (previousRecall - recall) * maxPrecision
        newRecall = recall
        newMaxPrecision = T.maximum(maxPrecision, precision)
        
        # Determine if the new or the old values should be returned
        areaUnderCurveToReturn = T.switch(T.eq(Y[index], 1), 
                                          newAreaUnderCurve, areaUnderCurve)
        recallToReturn = T.switch(T.eq(Y[index], 1), newRecall, previousRecall)
        maxPrecisionToReturn = T.switch(T.eq(Y[index], 1), 
                                        newMaxPrecision, maxPrecision)
        
        return areaUnderCurveToReturn, recallToReturn, maxPrecisionToReturn
        
    
    outputs, updates = theano.scan(fn=innerFunction,
                                    sequences = T.arange(Y_hat.shape[0]),
                                    non_sequences=[Y,Y_hat],
                                    outputs_info=[T.as_tensor_variable(0.0),
                                                  T.as_tensor_variable(0.0),
                                                  T.as_tensor_variable(0.0)],)
                                    
    area, prec, rec = outputs
    
    return area[-1] + rec[-1] * prec[-1]
    
