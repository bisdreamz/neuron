package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.ElementwiseDivide;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of ElementwiseDivide.
 * This class is only loaded when Vector API is available.
 */
public final class ElementwiseDivideVector implements ElementwiseDivide.Impl {
    
    @Override
    public void compute(float[] numerator, float[] denominator, float[] output) {
        if (!Vectorization.shouldVectorize(numerator.length)) {
            scalarCompute(numerator, denominator, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int upperBound = Vectorization.loopBound(numerator.length);
        
        int i = 0;
        for (; i < upperBound; i += species.length()) {
            FloatVector numVec = FloatVector.fromArray(species, numerator, i);
            FloatVector denVec = FloatVector.fromArray(species, denominator, i);
            
            numVec.div(denVec).intoArray(output, i);
        }
        
        for (; i < numerator.length; i++) {
            output[i] = numerator[i] / denominator[i];
        }
    }
    
    private void scalarCompute(float[] numerator, float[] denominator, float[] output) {
        for (int i = 0; i < numerator.length; i++) {
            output[i] = numerator[i] / denominator[i];
        }
    }
}