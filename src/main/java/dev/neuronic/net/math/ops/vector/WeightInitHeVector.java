package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.WeightInitHe;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of WeightInitHe.
 * This class is only loaded when Vector API is available.
 */
public final class WeightInitHeVector implements WeightInitHe.Impl {
    
    @Override
    public void compute(float[][] weights, int fanIn, FastRandom random) {
        compute(weights, fanIn, 0.0f, random);
    }

    @Override
    public void compute(float[][] weights, int fanIn, float noiseLevel, FastRandom random) {
        if (fanIn <= 0)
            throw new IllegalArgumentException("fanIn must be positive, got: " + fanIn);
        
        float scale = (float) Math.sqrt(2.0 / fanIn);
        
        if (shouldUseVectorized(weights))
            computeVectorized(weights, scale, noiseLevel, random);
        else
            computeScalar(weights, scale, noiseLevel, random);
    }
    
    private boolean shouldUseVectorized(float[][] weights) {
        for (float[] row : weights) {
            if (Vectorization.shouldVectorize(row.length))
                return true;
        }
        return false;
    }

    private void computeVectorized(float[][] weights, float scale, float noiseLevel, FastRandom random) {
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector scaleV = FloatVector.broadcast(species, scale);
        float[] tmp = new float[species.length()];
        
        for (float[] row : weights) {
            if (Vectorization.shouldVectorize(row.length))
                computeRowVectorized(row, scaleV, tmp, random, noiseLevel);
            else
                computeRowScalar(row, scale, random, noiseLevel);
        }
    }
    
    private void computeScalar(float[][] weights, float scale, float noiseLevel, FastRandom random) {
        for (float[] row : weights)
            computeRowScalar(row, scale, random, noiseLevel);
    }
    
    private void computeRowVectorized(float[] row, FloatVector scaleV, 
                                           float[] tmp, FastRandom random, float noiseLevel) {
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int len = row.length;
        int upper = Vectorization.loopBound(len);
        float scale = scaleV.lane(0);
        
        int i = 0;
        for (; i < upper; i += species.length()) {
            random.fillGaussian(tmp, 0f, 1f);
            float[] noise = new float[species.length()];
            random.fillUniform(noise, -0.5f, 0.5f);
            
            FloatVector.fromArray(species, tmp, 0)
                    .mul(scaleV)
                    .add(FloatVector.fromArray(species, noise, 0).mul(noiseLevel))
                    .intoArray(row, i);
        }

        for (; i < len; i++)
            row[i] = (float)(random.nextGaussian() * scale) + (random.nextFloat() - 0.5f) * noiseLevel;
    }

    private void computeRowScalar(float[] row, float scale, FastRandom random, float noiseLevel) {
        for (int i = 0; i < row.length; i++)
            row[i] = (float)(random.nextGaussian() * scale) + (random.nextFloat() - 0.5f) * noiseLevel;
    }
}