package dev.neuronic.net.optimizers;

import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.math.Parallelization;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;

/**
 * Adam (Adaptive Moment Estimation) optimizer with self-adaptive learning rates.
 * 
 * <p><b>⚠️ DEPRECATED:</b> Consider using {@link AdamWOptimizer} instead for modern deep learning applications.
 * AdamW provides better generalization through decoupled weight decay and is now the industry standard.
 * 
 * <p><b>What is Adam?</b> Adam is a popular optimization algorithm that combines the best
 * aspects of momentum and adaptive learning rates. It maintains a moving average of both
 * gradients (momentum) and squared gradients (adaptive rates) to automatically adjust
 * the learning rate for each parameter individually.
 * 
 * <p><b>Why use AdamW instead of Adam?</b>
 * <ul>
 *   <li><b>Better generalization:</b> Decoupled weight decay prevents overfitting more effectively</li>
 *   <li><b>Industry standard:</b> Used in all major transformer models (GPT, BERT, LLaMA)</li>
 *   <li><b>Theoretical soundness:</b> Fixes fundamental issues with Adam's L2 regularization</li>
 *   <li><b>Research consensus:</b> Default choice in modern deep learning papers</li>
 * </ul>
 * 
 * <p><b>When to still use Adam:</b>
 * <ul>
 *   <li><b>Legacy compatibility:</b> Reproducing results from older papers/models</li>
 *   <li><b>No regularization needed:</b> When weight decay is not beneficial</li>
 *   <li><b>Educational purposes:</b> Understanding the evolution from Adam to AdamW</li>
 * </ul>
 * 
 * <p><b>Recommended alternatives:</b>
 * <ul>
 *   <li><b>Modern choice:</b> {@link AdamWOptimizer} for most applications</li>
 *   <li><b>Simple models:</b> {@link SgdOptimizer} for linear regression, shallow networks</li>
 *   <li><b>Large batches:</b> SGD with momentum for very large batch training</li>
 * </ul>
 * 
 * <p><b>Adam Algorithm:</b>
 * <pre>
 * m_t = β₁ * m_{t-1} + (1 - β₁) * gradient    // Momentum (moving average of gradients)
 * v_t = β₂ * v_{t-1} + (1 - β₂) * gradient²   // Velocity (moving average of squared gradients)
 * m̂_t = m_t / (1 - β₁^t)                      // Bias correction for momentum
 * v̂_t = v_t / (1 - β₂^t)                      // Bias correction for velocity
 * param = param - α * m̂_t / (√v̂_t + ε)        // Parameter update
 * </pre>
 * 
 * <p><b>Parameter recommendations:</b>
 * <ul>
 *   <li><b>Learning rate (α):</b> 0.001 (good default), try 0.003 or 0.0003 if needed</li>
 *   <li><b>β₁ (momentum):</b> 0.9 (almost always good)</li>
 *   <li><b>β₂ (velocity):</b> 0.999 (almost always good)</li>
 *   <li><b>ε (epsilon):</b> 1e-8 (prevents division by zero)</li>
 * </ul>
 */
@Deprecated(forRemoval = false)
public class AdamOptimizer implements Optimizer, Serializable {
    
    private volatile float learningRate; // Made volatile for thread-safe updates
    private final float beta1;        // Momentum decay rate
    private final float beta2;        // Velocity decay rate  
    private final float epsilon;      // Small constant to avoid division by zero
    
    // Thread-safe state storage per layer (identified by weights reference)
    private final ConcurrentHashMap<Object, AdamState> layerStates = new ConcurrentHashMap<>();
    
    /**
     * Create Adam optimizer with default parameters.
     * 
     * <p>Uses the widely successful default values:
     * <ul>
     *   <li>β₁ = 0.9 (momentum)</li>
     *   <li>β₂ = 0.999 (velocity)</li>
     *   <li>ε = 1e-8 (epsilon)</li>
     * </ul>
     *
     * @see AdamWOptimizer instead
     * 
     * @param learningRate the learning rate (typically 0.001, try 0.003 or 0.0003)
     */
    @Deprecated
    public AdamOptimizer(float learningRate) {
        this(learningRate, 0.9f, 0.999f, 1e-8f);
    }
    
    /**
     * Create Adam optimizer with custom parameters.
     * 
     * <p><b>Parameter guidance:</b>
     * <ul>
     *   <li><b>β₁ closer to 1:</b> More momentum, smoother updates</li>
     *   <li><b>β₁ closer to 0:</b> Less momentum, more responsive to recent gradients</li>
     *   <li><b>β₂ closer to 1:</b> More adaptive, larger effective learning rates for sparse parameters</li>
     *   <li><b>β₂ closer to 0:</b> Less adaptive, more like momentum SGD</li>
     * </ul>
     * 
     * @param learningRate the learning rate (typically 0.001)
     * @param beta1 momentum decay rate (typically 0.9)
     * @param beta2 velocity decay rate (typically 0.999)
     * @param epsilon small constant to avoid division by zero (typically 1e-8)
     */
    public AdamOptimizer(float learningRate, float beta1, float beta2, float epsilon) {
        if (learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive: " + learningRate);
        if (beta1 < 0 || beta1 >= 1)
            throw new IllegalArgumentException("Beta1 must be in [0, 1): " + beta1);
        if (beta2 < 0 || beta2 >= 1)
            throw new IllegalArgumentException("Beta2 must be in [0, 1): " + beta2);
        if (epsilon <= 0)
            throw new IllegalArgumentException("Epsilon must be positive: " + epsilon);
            
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }
    
    @Override
    public void optimize(Object stateKey, float[][] weights, float[] biases, float[][] weightGradients,
                        float[] biasGradients, ExecutorService executor) {
        // Get or create state for this layer using the stable stateKey
        AdamState state = layerStates.computeIfAbsent(stateKey, k -> new AdamState(weights, biases));

        // Increment time step atomically and capture the value
        long currentTimeStep = state.timeStep.incrementAndGet();

        // Check if we should parallelize
        if (executor != null && Parallelization.shouldParallelize(weights.length, executor)) {
            parallelOptimize(weights, biases, weightGradients, biasGradients, state, currentTimeStep, executor);
        } else {
            sequentialOptimize(weights, biases, weightGradients, biasGradients, state, currentTimeStep);
        }
    }

    @Override
    public void optimize(float[][] weights, float[] biases, float[][] weightGradients, float[] biasGradients) {
        // Fallback for dense layers: the weights array itself is the stable key
        optimize(weights, weights, biases, weightGradients, biasGradients, null);
    }

    @Override
    public void optimize(float[][] weights, float[] biases, float[][] weightGradients, 
                        float[] biasGradients, ExecutorService executor) {
        // Fallback for dense layers: the weights array itself is the stable key
        optimize(weights, weights, biases, weightGradients, biasGradients, executor);
    }
    
    /**
     * Parallel optimization across weight matrix rows.
     */
    private void parallelOptimize(float[][] weights, float[] biases, float[][] weightGradients, 
                                 float[] biasGradients, AdamState state, long currentTimeStep, ExecutorService executor) {
        float momentumCorrection = 1.0f - (float) Math.pow(beta1, currentTimeStep);
        float velocityCorrection = 1.0f - (float) Math.pow(beta2, currentTimeStep);
        
        // Calculate optimal thread count and work distribution
        int numThreads = Parallelization.calculateOptimalThreads(weights.length, executor);
        Parallelization.WorkRange[] ranges =
            Parallelization.splitWork(weights.length, numThreads);
        
        // Create parallel tasks for weight updates
        Runnable[] weightTasks = new Runnable[ranges.length];
        for (int t = 0; t < ranges.length; t++) {
            final int threadId = t;
            weightTasks[t] = () -> {
                Parallelization.WorkRange range = ranges[threadId];
                for (int i = range.start; i < range.end; i++) {
                    adamUpdateWithPrecomputedCorrectionAndBuffers(weights[i], weightGradients[i], 
                        state.weightMomentum[i], state.weightVelocity[i], 
                        momentumCorrection, velocityCorrection, state.getWeightBuffers());
                }
            };
        }
        
        // Execute weight updates in parallel
        Parallelization.executeParallel(executor, weightTasks);
        
        // Update biases sequentially (usually small arrays, not worth parallelizing)
        adamUpdateWithPrecomputedCorrectionAndBuffers(biases, biasGradients, state.biasMomentum, 
                                           state.biasVelocity, momentumCorrection, velocityCorrection, state.getBiasBuffers());
    }
    
    /**
     * Sequential optimization (fallback when parallelization isn't beneficial).
     */
    private void sequentialOptimize(float[][] weights, float[] biases, float[][] weightGradients, 
                                   float[] biasGradients, AdamState state, long currentTimeStep) {
        // Pre-compute bias correction factors (avoid redundant computation)
        float momentumCorrection = 1.0f - (float) Math.pow(beta1, currentTimeStep);
        float velocityCorrection = 1.0f - (float) Math.pow(beta2, currentTimeStep);
        
        // Update weights using reusable buffers
        for (int i = 0; i < weights.length; i++) {
            adamUpdateWithPrecomputedCorrectionAndBuffers(weights[i], weightGradients[i], 
                state.weightMomentum[i], state.weightVelocity[i], 
                momentumCorrection, velocityCorrection, state.getWeightBuffers());
        }
        
        // Update biases using reusable buffers
        adamUpdateWithPrecomputedCorrectionAndBuffers(biases, biasGradients, state.biasMomentum, 
            state.biasVelocity, momentumCorrection, velocityCorrection, state.getBiasBuffers());
    }
    
    
    /**
     * Core Adam update algorithm with precomputed bias correction factors.
     * Used for parallel optimization to avoid redundant computations.
     * 
     * @deprecated Use adamUpdateWithPrecomputedCorrectionAndBuffers for zero-allocation performance
     */
    private void adamUpdateWithPrecomputedCorrection(float[] params, float[] gradients, 
                                                    float[] momentum, float[] velocity,
                                                    float momentumCorrection, float velocityCorrection) {
        // This method is deprecated but kept for compatibility
        // It still allocates temporary arrays - use the buffered version instead
        throw new UnsupportedOperationException("Use adamUpdateWithPrecomputedCorrectionAndBuffers for zero-allocation performance");
    }
    
    /**
     * Memory-optimized Adam update using reusable buffers.
     * Eliminates all temporary array allocations during optimization.
     * All operations are fully vectorized for maximum performance.
     * 
     * @param buffers Reusable buffer array: [gradientSquared, biascorrectedMomentum, biascorrectedVelocity, sqrtVelocity]
     */
    private void adamUpdateWithPrecomputedCorrectionAndBuffers(float[] params, float[] gradients, 
                                                              float[] momentum, float[] velocity,
                                                              float momentumCorrection, float velocityCorrection,
                                                              float[][] buffers) {
        // Extract reusable buffers (no allocations!)
        float[] gradientSquared = buffers[0];
        float[] biascorrectedMomentum = buffers[1];
        float[] biascorrectedVelocity = buffers[2];
        float[] sqrtVelocity = buffers[3];
        
        // Update momentum: m_t = β₁ * m_{t-1} + (1 - β₁) * gradient (VECTORIZED)
        NetMath.exponentialMovingAverageInPlace(momentum, gradients, beta1);
        
        // Update velocity: v_t = β₂ * v_{t-1} + (1 - β₂) * gradient² (VECTORIZED)
        NetMath.elementwiseSquare(gradients, gradientSquared);
        NetMath.exponentialMovingAverageInPlace(velocity, gradientSquared, beta2);
        
        // Apply bias-corrected updates: param = param - α * m̂_t / (√v̂_t + ε) (VECTORIZED)
        NetMath.elementwiseScale(momentum, 1.0f / momentumCorrection, biascorrectedMomentum);
        NetMath.elementwiseScale(velocity, 1.0f / velocityCorrection, biascorrectedVelocity);
        NetMath.elementwiseSqrtWithEpsilon(biascorrectedVelocity, epsilon, sqrtVelocity);
        
        // Final parameter update (VECTORIZED): params -= learningRate * (momentum / sqrtVelocity)
        NetMath.fusedMultiplyDivideSubtract(params, biascorrectedMomentum, sqrtVelocity, learningRate);
    }
    
    /**
     * State storage for Adam optimizer per layer.
     * Thread-safe through ConcurrentHashMap access patterns.
     * Includes reusable buffers to eliminate memory allocations during updates.
     */
    private static class AdamState {
        final float[][] weightMomentum;    // m_t for weights
        final float[][] weightVelocity;    // v_t for weights  
        final float[] biasMomentum;        // m_t for biases
        final float[] biasVelocity;        // v_t for biases
        final java.util.concurrent.atomic.AtomicLong timeStep = new java.util.concurrent.atomic.AtomicLong(0); // t (for bias correction)
        
        // Reusable buffers per weight row (ThreadLocal for thread safety)
        final ThreadLocal<float[][]> weightBuffers;
        final ThreadLocal<float[][]> biasBuffers;
        
        AdamState(float[][] weights, float[] biases) {
            // Initialize momentum and velocity to zeros
            weightMomentum = new float[weights.length][];
            weightVelocity = new float[weights.length][];
            for (int i = 0; i < weights.length; i++) {
                weightMomentum[i] = new float[weights[i].length];
                weightVelocity[i] = new float[weights[i].length];
            }
            
            biasMomentum = new float[biases.length];
            biasVelocity = new float[biases.length];
            
            // Initialize reusable buffers
            // Each thread gets its own set of buffers to avoid contention
            weightBuffers = ThreadLocal.withInitial(() -> {
                int maxLength = 0;
                for (float[] row : weights) {
                    maxLength = Math.max(maxLength, row.length);
                }
                return new float[][] {
                    new float[maxLength], // gradientSquared buffer
                    new float[maxLength], // biascorrectedMomentum buffer  
                    new float[maxLength], // biascorrectedVelocity buffer
                    new float[maxLength]  // sqrtVelocity buffer
                };
            });
            
            biasBuffers = ThreadLocal.withInitial(() -> new float[][] {
                new float[biases.length], // gradientSquared buffer
                new float[biases.length], // biascorrectedMomentum buffer
                new float[biases.length], // biascorrectedVelocity buffer  
                new float[biases.length]  // sqrtVelocity buffer
            });
        }
        
        /**
         * Get reusable buffers for weight updates (thread-safe).
         * Returns [gradientSquared, biascorrectedMomentum, biascorrectedVelocity, sqrtVelocity]
         */
        float[][] getWeightBuffers() {
            return weightBuffers.get();
        }
        
        /**
         * Get reusable buffers for bias updates (thread-safe).
         * Returns [gradientSquared, biascorrectedMomentum, biascorrectedVelocity, sqrtVelocity]
         */
        float[][] getBiasBuffers() {
            return biasBuffers.get();
        }
    }
    
    /**
     * @return the learning rate for this optimizer
     */
    public float getLearningRate() {
        return learningRate;
    }
    
    /**
     * @return the momentum decay rate (β₁)
     */
    public float getBeta1() {
        return beta1;
    }
    
    /**
     * @return the velocity decay rate (β₂)
     */
    public float getBeta2() {
        return beta2;
    }
    
    /**
     * @return the epsilon value
     */
    public float getEpsilon() {
        return epsilon;
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        out.writeFloat(learningRate);
        out.writeFloat(beta1);
        out.writeFloat(beta2);
        out.writeFloat(epsilon);
        // Note: We don't serialize the state - it will be rebuilt during training
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize(DataInputStream, int) static method instead");
    }
    
    /**
     * Static method to deserialize an AdamOptimizer from stream.
     */
    public static AdamOptimizer deserialize(DataInputStream in, int version) throws IOException {
        float learningRate = in.readFloat();
        float beta1 = in.readFloat();
        float beta2 = in.readFloat();
        float epsilon = in.readFloat();
        return new AdamOptimizer(learningRate, beta1, beta2, epsilon);
    }
    
    @Override
    public int getSerializedSize(int version) {
        return 16; // 4 floats
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_ADAM_OPTIMIZER;
    }
    
    @Override
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void sparseOptimize(Object stateKey, float[][] allWeights, int[] indicesToUpdate,
                               float[][] gradients, ExecutorService executor) {
        if (indicesToUpdate.length != gradients.length) {
            throw new IllegalArgumentException(String.format(
                "Mismatched inputs for sparse update: %d indices but %d gradients.",
                indicesToUpdate.length, gradients.length));
        }
        if (indicesToUpdate.length == 0) {
            return; // Nothing to do
        }

        // Get or create state for this layer using the stable stateKey.
        AdamState state = layerStates.computeIfAbsent(stateKey, k -> new AdamState(allWeights, new float[0]));

        // Increment time step atomically
        long currentTimeStep = state.timeStep.incrementAndGet();

        // Pre-compute bias correction factors
        float momentumCorrection = 1.0f - (float) Math.pow(beta1, currentTimeStep);
        float velocityCorrection = 1.0f - (float) Math.pow(beta2, currentTimeStep);

        // Loop through only the touched indices
        for (int i = 0; i < indicesToUpdate.length; i++) {
            int weightIndex = indicesToUpdate[i];
            float[] gradient = gradients[i];

            if (weightIndex < 0 || weightIndex >= allWeights.length) {
                 System.err.printf("Optimizer Warning: Index %d is out of bounds for weights (len=%d). Skipping.\n",
                                  weightIndex, allWeights.length);
                continue;
            }
            if (weightIndex >= state.weightMomentum.length) {
                System.err.printf("CRITICAL OPTIMIZER ERROR: Index %d is out of bounds for momentum state (len=%d). " +
                                  "This likely means the stateKey is not being used correctly. Skipping update.\n",
                                  weightIndex, state.weightMomentum.length);
                continue;
            }

            // Perform the Adam update on the specific row
            adamUpdateWithPrecomputedCorrectionAndBuffers(
                allWeights[weightIndex], gradient,
                state.weightMomentum[weightIndex], state.weightVelocity[weightIndex],
                momentumCorrection, velocityCorrection, state.getWeightBuffers()
            );
        }
    }
}