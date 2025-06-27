package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.LeakyReLU;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of LeakyReLU.
 * This class is only loaded when Vector API is available.
 */
public final class LeakyReLUVector implements LeakyReLU.Impl {
    
    @Override
    public void activate(float[] input, float alpha, float[] output) {
        if (!Vectorization.shouldVectorize(input.length)) {
            scalarActivate(input, alpha, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector zero = FloatVector.zero(species);
        FloatVector alphaVec = FloatVector.broadcast(species, alpha);
        int length = input.length;
        int L = species.length();
        int upper = Vectorization.loopBound(length);

        for (int i = 0; i < upper; i += L) {
            FloatVector v = FloatVector.fromArray(species, input, i);
            VectorMask<Float> positive = v.compare(VectorOperators.GT, zero);
            FloatVector negativeResult = v.mul(alphaVec);
            negativeResult.blend(v, positive).intoArray(output, i);
        }

        for (int i = upper; i < length; i++)
            output[i] = input[i] > 0f ? input[i] : input[i] * alpha;
    }
    
    @Override
    public void derivative(float[] input, float alpha, float[] output) {
        if (!Vectorization.shouldVectorize(input.length)) {
            scalarDerivative(input, alpha, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector zero = FloatVector.zero(species);
        FloatVector one = FloatVector.broadcast(species, 1.0f);
        FloatVector alphaVec = FloatVector.broadcast(species, alpha);
        int length = input.length;
        int L = species.length();
        int upper = Vectorization.loopBound(length);

        for (int i = 0; i < upper; i += L) {
            FloatVector v = FloatVector.fromArray(species, input, i);
            VectorMask<Float> positive = v.compare(VectorOperators.GT, zero);
            alphaVec.blend(one, positive).intoArray(output, i);
        }

        for (int i = upper; i < length; i++)
            output[i] = input[i] > 0f ? 1.0f : alpha;
    }
    
    @Override
    public void activateRange(float[] input, float alpha, float[] output, int start, int end) {
        if (!Vectorization.shouldVectorize(end - start)) {
            scalarActivateRange(input, alpha, output, start, end);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector zero = FloatVector.zero(species);
        FloatVector alphaVec = FloatVector.broadcast(species, alpha);
        int L = species.length();
        int upper = start + ((end - start) / L) * L;
        
        for (int i = start; i < upper; i += L) {
            FloatVector v = FloatVector.fromArray(species, input, i);
            VectorMask<Float> positive = v.compare(VectorOperators.GT, zero);
            FloatVector negativeResult = v.mul(alphaVec);
            negativeResult.blend(v, positive).intoArray(output, i);
        }
        
        for (int i = upper; i < end; i++)
            output[i] = input[i] > 0f ? input[i] : input[i] * alpha;
    }
    
    @Override
    public void derivativeRange(float[] input, float alpha, float[] output, int start, int end) {
        if (!Vectorization.shouldVectorize(end - start)) {
            scalarDerivativeRange(input, alpha, output, start, end);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector zero = FloatVector.zero(species);
        FloatVector one = FloatVector.broadcast(species, 1.0f);
        FloatVector alphaVec = FloatVector.broadcast(species, alpha);
        int L = species.length();
        int upper = start + ((end - start) / L) * L;
        
        for (int i = start; i < upper; i += L) {
            FloatVector v = FloatVector.fromArray(species, input, i);
            VectorMask<Float> positive = v.compare(VectorOperators.GT, zero);
            alphaVec.blend(one, positive).intoArray(output, i);
        }
        
        for (int i = upper; i < end; i++)
            output[i] = input[i] > 0f ? 1.0f : alpha;
    }
    
    private void scalarActivate(float[] input, float alpha, float[] output) {
        for (int i = 0; i < input.length; i++)
            output[i] = input[i] > 0f ? input[i] : input[i] * alpha;
    }
    
    private void scalarDerivative(float[] input, float alpha, float[] output) {
        for (int i = 0; i < input.length; i++)
            output[i] = input[i] > 0f ? 1.0f : alpha;
    }
    
    private void scalarActivateRange(float[] input, float alpha, float[] output, int start, int end) {
        for (int i = start; i < end; i++)
            output[i] = input[i] > 0f ? input[i] : input[i] * alpha;
    }
    
    private void scalarDerivativeRange(float[] input, float alpha, float[] output, int start, int end) {
        for (int i = start; i < end; i++)
            output[i] = input[i] > 0f ? 1.0f : alpha;
    }
}