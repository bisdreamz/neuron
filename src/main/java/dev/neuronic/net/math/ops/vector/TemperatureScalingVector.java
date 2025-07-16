package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.TemperatureScaling;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of TemperatureScaling.
 * This class is only loaded when Vector API is available.
 */
public final class TemperatureScalingVector implements TemperatureScaling.Impl {
    
    private static final float LOG_EPSILON = -1e10f;
    
    @Override
    public void apply(float[] probabilities, float temperature, float[] output) {
        // Check if vectorization is worthwhile for this array size
        if (!Vectorization.shouldVectorizeLimited(probabilities.length, 64)) {
            // Fall back to scalar for small arrays
            TemperatureScaling.applyScalar(probabilities, temperature, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        final int length = probabilities.length;
        final int lanes = species.length();
        final int loopBound = Vectorization.loopBound(length);
        
        FloatVector tempVec = FloatVector.broadcast(species, temperature);
        FloatVector logEpsilonVec = FloatVector.broadcast(species, LOG_EPSILON / temperature);
        
        // Step 1: Convert to log-space and apply temperature, find max
        float maxLogit = Float.NEGATIVE_INFINITY;
        
        // Vectorized portion
        for (int i = 0; i < loopBound; i += lanes) {
            FloatVector probVec = FloatVector.fromArray(species, probabilities, i);
            
            // Create mask for positive values
            var mask = probVec.compare(VectorOperators.GT, 0.0f);
            
            // Compute log for positive values
            FloatVector logVec = probVec.lanewise(VectorOperators.LOG, mask);
            
            // Divide by temperature
            FloatVector scaledVec = logVec.div(tempVec);
            
            // For non-positive values, use LOG_EPSILON / temperature
            scaledVec = scaledVec.blend(logEpsilonVec, mask.not());
            
            // Store result
            scaledVec.intoArray(output, i);
            
            // Update max
            float localMax = scaledVec.reduceLanes(VectorOperators.MAX);
            maxLogit = Math.max(maxLogit, localMax);
        }
        
        // Scalar tail and final max computation
        for (int i = loopBound; i < length; i++) {
            if (probabilities[i] > 0) {
                output[i] = (float) Math.log(probabilities[i]) / temperature;
            } else {
                output[i] = LOG_EPSILON / temperature;
            }
            maxLogit = Math.max(maxLogit, output[i]);
        }
        
        // Step 2: Apply exp(x - max) and compute sum
        FloatVector maxVec = FloatVector.broadcast(species, maxLogit);
        FloatVector sumVec = FloatVector.zero(species);
        
        // Vectorized exp and sum
        for (int i = 0; i < loopBound; i += lanes) {
            FloatVector logitVec = FloatVector.fromArray(species, output, i);
            FloatVector expVec = logitVec.sub(maxVec).lanewise(VectorOperators.EXP);
            expVec.intoArray(output, i);
            sumVec = sumVec.add(expVec);
        }
        
        // Scalar tail
        float sumExp = sumVec.reduceLanes(VectorOperators.ADD);
        for (int i = loopBound; i < length; i++) {
            output[i] = (float) Math.exp(output[i] - maxLogit);
            sumExp += output[i];
        }
        
        // Step 3: Normalize
        FloatVector sumExpVec = FloatVector.broadcast(species, sumExp);
        
        for (int i = 0; i < loopBound; i += lanes) {
            FloatVector vec = FloatVector.fromArray(species, output, i);
            vec.div(sumExpVec).intoArray(output, i);
        }
        
        // Scalar tail for normalization
        for (int i = loopBound; i < length; i++) {
            output[i] /= sumExp;
        }
    }
}