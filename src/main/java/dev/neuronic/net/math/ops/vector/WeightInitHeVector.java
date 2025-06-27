package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.WeightInitHe;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import java.util.random.RandomGenerator;

/**
 * Vector implementation of WeightInitHe.
 * This class is only loaded when Vector API is available.
 */
public final class WeightInitHeVector implements WeightInitHe.Impl {
    
    @Override
    public void compute(float[][] weights, int fanIn) {
        if (fanIn <= 0)
            throw new IllegalArgumentException("fanIn must be positive, got: " + fanIn);
        
        float scale = (float) Math.sqrt(2.0 / fanIn);
        
        if (shouldUseVectorized(weights))
            computeVectorized(weights, scale);
        else
            computeScalar(weights, scale);
    }
    
    private boolean shouldUseVectorized(float[][] weights) {
        for (float[] row : weights) {
            if (Vectorization.shouldVectorize(row.length))
                return true;
        }
        return false;
    }

    private void computeVectorized(float[][] weights, float scale) {
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector scaleV = FloatVector.broadcast(species, scale);
        float[] tmp = new float[species.length()];
        RandomGenerator rnd = FastRandom.get();
        
        for (float[] row : weights) {
            if (Vectorization.shouldVectorize(row.length))
                computeRowVectorized(row, scaleV, tmp, rnd);
            else
                computeRowScalar(row, scale, rnd);
        }
    }
    
    private void computeScalar(float[][] weights, float scale) {
        RandomGenerator rnd = FastRandom.get();
        
        for (float[] row : weights)
            computeRowScalar(row, scale, rnd);
    }
    
    private void computeRowVectorized(float[] row, FloatVector scaleV, 
                                           float[] tmp, RandomGenerator rnd) {
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int len = row.length;
        int upper = Vectorization.loopBound(len);
        float scale = scaleV.lane(0);
        
        int i = 0;
        for (; i < upper; i += species.length()) {
            FastRandom.fillGaussian(tmp, 0f, 1f);
            
            FloatVector.fromArray(species, tmp, 0)
                    .mul(scaleV)
                    .intoArray(row, i);
        }

        for (; i < len; i++)
            row[i] = (float)(rnd.nextGaussian() * scale);
    }

    private void computeRowScalar(float[] row, float scale, RandomGenerator rnd) {
        for (int i = 0; i < row.length; i++)
            row[i] = (float)(rnd.nextGaussian() * scale);
    }
}