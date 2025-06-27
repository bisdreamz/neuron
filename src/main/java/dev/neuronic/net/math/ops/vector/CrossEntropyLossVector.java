package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.CrossEntropyLoss;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of CrossEntropyLoss.
 * This class is only loaded when Vector API is available.
 */
public final class CrossEntropyLossVector implements CrossEntropyLoss.Impl {
    
    @Override
    public float compute(float[] trueLabels, float[] predictions) {
        if (!Vectorization.shouldVectorize(trueLabels.length)) {
            return scalarCompute(trueLabels, predictions);
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int length = trueLabels.length;
        int upperBound = Vectorization.loopBound(length);
        
        FloatVector sumVector = FloatVector.zero(species);
        
        for (int i = 0; i < upperBound; i += species.length()) {
            FloatVector trueVec = FloatVector.fromArray(species, trueLabels, i);
            FloatVector predVec = FloatVector.fromArray(species, predictions, i);
            
            predVec = predVec.max(FloatVector.broadcast(species, 1e-7f));
            
            FloatVector logPred = predVec.lanewise(VectorOperators.LOG);
            FloatVector product = trueVec.mul(logPred);
            sumVector = sumVector.add(product);
        }
        
        float sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (int i = upperBound; i < length; i++) {
            float pred = Math.max(predictions[i], 1e-7f);
            sum += trueLabels[i] * Math.log(pred);
        }
        
        return -sum;
    }
    
    @Override
    public void gradient(float[] trueLabels, float[] predictions, float[] output) {
        if (!Vectorization.shouldVectorize(trueLabels.length)) {
            scalarGradient(trueLabels, predictions, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int length = trueLabels.length;
        int upperBound = Vectorization.loopBound(length);
        
        for (int i = 0; i < upperBound; i += species.length()) {
            FloatVector trueVec = FloatVector.fromArray(species, trueLabels, i);
            FloatVector predVec = FloatVector.fromArray(species, predictions, i);
            
            predVec.sub(trueVec).intoArray(output, i);
        }
        
        for (int i = upperBound; i < length; i++) {
            output[i] = predictions[i] - trueLabels[i];
        }
    }
    
    private float scalarCompute(float[] trueLabels, float[] predictions) {
        float sum = 0.0f;
        for (int i = 0; i < trueLabels.length; i++) {
            float pred = Math.max(predictions[i], 1e-7f);
            sum += trueLabels[i] * Math.log(pred);
        }
        return -sum;
    }
    
    private void scalarGradient(float[] trueLabels, float[] predictions, float[] output) {
        for (int i = 0; i < trueLabels.length; i++) {
            output[i] = predictions[i] - trueLabels[i];
        }
    }
}