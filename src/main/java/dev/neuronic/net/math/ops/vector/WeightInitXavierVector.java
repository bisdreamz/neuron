package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.WeightInitXavier;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import java.util.random.RandomGenerator;

/**
 * Vector implementation of WeightInitXavier.
 * This class is only loaded when Vector API is available.
 */
public final class WeightInitXavierVector implements WeightInitXavier.Impl {
    
    @Override
    public void compute(float[][] weights, int fanIn, int fanOut) {
        if (fanIn <= 0 || fanOut <= 0)
            throw new IllegalArgumentException("fanIn and fanOut must be positive, got: " + fanIn + ", " + fanOut);
            
        float limit = (float)Math.sqrt(6.0f / (fanIn + fanOut));
        
        if (shouldUseVectorized(weights))
            computeVectorized(weights, limit);
        else
            computeScalar(weights, limit);
    }
    
    private boolean shouldUseVectorized(float[][] weights) {
        for (float[] row : weights) {
            if (Vectorization.shouldVectorize(row.length))
                return true;
        }
        return false;
    }

    private void computeVectorized(float[][] weights, float limit) {
        VectorSpecies<Float> species = Vectorization.getSpecies();
        float twoLimit = 2f * limit;
        FloatVector limitVec = FloatVector.broadcast(species, limit);
        FloatVector twoLimitVec = FloatVector.broadcast(species, twoLimit);
        float[] tmp = new float[species.length()];
        
        RandomGenerator rnd = FastRandom.get();
        
        for (float[] row : weights) {
            if (Vectorization.shouldVectorize(row.length))
                computeRowVectorized(row, limitVec, twoLimitVec, tmp);
            else
                computeRowScalar(row, twoLimit, rnd);
        }
    }
    
    private void computeScalar(float[][] weights, float limit) {
        float twoLimit = 2f * limit;
        RandomGenerator rnd = FastRandom.get();
        
        for (float[] row : weights)
            computeRowScalar(row, twoLimit, rnd);
    }
    
    private void computeRowVectorized(float[] row, FloatVector limitVec, 
                                           FloatVector twoLimitVec, float[] tmp) {
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int upper = Vectorization.loopBound(row.length);

        for (int i = 0; i < upper; i += species.length()) {
            FastRandom.fillUniform(tmp, 0f, 1f);
            FloatVector.fromArray(species, tmp, 0)
                    .mul(twoLimitVec)
                    .sub(limitVec)
                    .intoArray(row, i);
        }

        RandomGenerator rnd = FastRandom.get();
        float twoLimit = 2f * limitVec.lane(0);
        for (int i = upper; i < row.length; i++) {
            row[i] = (rnd.nextFloat() - 0.5f) * twoLimit;
        }
    }

    private void computeRowScalar(float[] row, float twoLimit, RandomGenerator rnd) {
        for (int i = 0; i < row.length; i++)
            row[i] = (rnd.nextFloat() - 0.5f) * twoLimit;
    }
}