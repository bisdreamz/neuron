package dev.neuronic.net.optimizers;

import dev.neuronic.net.math.Parallelization;
import dev.neuronic.net.optimizers.adamw.FusedAdamWUpdate;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicLong;

/**
 * AdamW optimizer with decoupled weight decay for improved generalization.
 * 
 * <p><b>What is AdamW?</b> AdamW is an improved version of Adam that fixes how weight
 * decay is applied. While standard Adam adds L2 penalty to gradients, AdamW applies
 * weight decay directly to parameters. This "decoupled" approach is more effective
 * and predictable, especially with adaptive learning rate optimizers.
 * 
 * <p><b>Adam vs AdamW - The Key Difference:</b>
 * <ul>
 *   <li><b>Adam:</b> Applies L2 penalty to gradients, then adapts learning rate</li>
 *   <li><b>AdamW:</b> Adapts learning rate first, then applies weight decay directly</li>
 * </ul>
 * This seemingly small change makes AdamW significantly more effective at regularization.
 * 
 * <p><b>Why use AdamW instead of Adam?</b>
 * <ul>
 *   <li><b>Better generalization:</b> More effective regularization prevents overfitting</li>
 *   <li><b>Predictable weight decay:</b> Weight decay effect doesn't interact with adaptive learning rates</li>
 *   <li><b>State-of-the-art results:</b> Used in modern transformer models (BERT, GPT, etc.)</li>
 *   <li><b>Cleaner separation:</b> Optimization and regularization are independent</li>
 * </ul>
 * 
 * <p><b>When to use AdamW:</b>
 * <ul>
 *   <li><b>Large models:</b> Transformers, large CNNs, any model prone to overfitting</li>
 *   <li><b>Limited data:</b> When you need strong regularization</li>
 *   <li><b>Production models:</b> When you care about generalization to real-world data</li>
 *   <li><b>Transfer learning:</b> Fine-tuning pre-trained models</li>
 * </ul>
 * 
 * <p><b>AdamW Algorithm:</b>
 * <pre>
 * m_t = β₁ * m_{t-1} + (1 - β₁) * gradient      // Momentum
 * v_t = β₂ * v_{t-1} + (1 - β₂) * gradient²     // Velocity  
 * m̂_t = m_t / (1 - β₁^t)                        // Bias correction
 * v̂_t = v_t / (1 - β₂^t)                        // Bias correction
 * param = param - α * m̂_t / (√v̂_t + ε)          // Adam update
 * param = param * (1 - λ)                       // Weight decay (decoupled)
 * </pre>
 * 
 * <p><b>Parameter recommendations:</b>
 * <ul>
 *   <li><b>Learning rate (α):</b> 0.001 (good default), try 0.003 or 0.0003</li>
 *   <li><b>Weight decay (λ):</b> 0.01 (typical), try 0.001-0.1 depending on model size</li>
 *   <li><b>β₁:</b> 0.9 (almost always good)</li>
 *   <li><b>β₂:</b> 0.999 (almost always good)</li>
 *   <li><b>ε:</b> 1e-8 (prevents division by zero)</li>
 * </ul>
 * 
 * <p><b>Weight decay guidance:</b>
 * <ul>
 *   <li><b>0.001:</b> Light regularization for large models or lots of data</li>
 *   <li><b>0.01:</b> Good default for most cases</li>
 *   <li><b>0.1:</b> Strong regularization for small models or limited data</li>
 * </ul>
 * 
 * <p><b>Automatic Embedding Optimization:</b>
 * When used with embedding layers, AdamW automatically adjusts parameters:
 * <ul>
 *   <li><b>5x higher learning rate:</b> Compensates for sparse updates on high-cardinality features</li>
 *   <li><b>10x less weight decay:</b> Prevents rare embeddings from being zeroed out</li>
 * </ul>
 * This follows best practices from Google, Facebook, and modern RecSys literature.
 */
public class AdamWOptimizer implements Optimizer, Serializable {

    private volatile float learningRate; // Made volatile for thread-safe updates
    private final float beta1;           // Momentum decay rate
    private final float beta2;           // Velocity decay rate
    private final float epsilon;         // Small constant to avoid division by zero
    private final float weightDecay;     // Weight decay coefficient
    
    // Thread-safe state storage per layer (identified by weights reference)
    private final ConcurrentHashMap<Object, AdamWState> layerStates = new ConcurrentHashMap<>();
    
    /**
     * Create AdamW optimizer with specified learning rate and weight decay.
     * 
     * <p>Uses proven default values for β₁, β₂, and ε while allowing you to
     * set the two most important hyperparameters: learning rate and weight decay.
     * 
     * @param learningRate the learning rate (typically 0.001 is a good starting point)
     * @param weightDecay weight decay coefficient (typically 0.01, use 0 to disable)
     */
    public AdamWOptimizer(float learningRate, float weightDecay) {
        this(learningRate, 0.9f, 0.999f, 1e-8f, weightDecay);
    }
    
    /**
     * Create AdamW optimizer with default parameters.
     * Uses learning rate 0.001 and weight decay 0.01 - good starting points for most models.
     */
    public AdamWOptimizer() {
        this(0.001f, 0.01f);
    }
    
    /**
     * Create AdamW optimizer with full parameter control.
     * 
     * @param learningRate the learning rate (typically 0.001)
     * @param beta1 momentum decay rate (typically 0.9)
     * @param beta2 velocity decay rate (typically 0.999)
     * @param epsilon small constant to avoid division by zero (typically 1e-8)
     * @param weightDecay weight decay coefficient (typically 0.01)
     */
    public AdamWOptimizer(float learningRate, float beta1, float beta2, float epsilon, float weightDecay) {
        if (learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive: " + learningRate);
        if (beta1 < 0 || beta1 >= 1)
            throw new IllegalArgumentException("Beta1 must be in [0, 1): " + beta1);
        if (beta2 < 0 || beta2 >= 1)
            throw new IllegalArgumentException("Beta2 must be in [0, 1): " + beta2);
        if (epsilon <= 0)
            throw new IllegalArgumentException("Epsilon must be positive: " + epsilon);
        if (weightDecay < 0)
            throw new IllegalArgumentException("Weight decay must be non-negative: " + weightDecay);
            
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }
    
    
    @Override
    public void optimize(Object stateKey, float[][] weights, float[] biases, float[][] weightGradients,
                        float[] biasGradients, ExecutorService executor) {
        // Get or create state for this layer using the stable stateKey
        AdamWState state = layerStates.computeIfAbsent(stateKey, k -> new AdamWState(weights, biases));

        // Increment time step atomically
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
                                 float[] biasGradients, AdamWState state, long currentTimeStep, ExecutorService executor) {
        // Pre-compute bias correction factors (shared across all threads)
        float momentumCorrection = 1.0f - (float) Math.pow(beta1, currentTimeStep);
        float velocityCorrection = 1.0f - (float) Math.pow(beta2, currentTimeStep);
        
        // Calculate optimal thread count and work distribution
        int numThreads = Parallelization.calculateOptimalThreads(weights.length, executor);
        Parallelization.WorkRange[] ranges = Parallelization.splitWork(weights.length, numThreads);
        
        // Create parallel tasks for weight updates
        Runnable[] weightTasks = new Runnable[ranges.length];
        for (int t = 0; t < ranges.length; t++) {
            final int threadId = t;
            weightTasks[t] = () -> {
                Parallelization.WorkRange range = ranges[threadId];
                for (int i = range.start; i < range.end; i++) {
                    FusedAdamWUpdate.compute(
                            weights[i], weightGradients[i],
                            state.weightMomentum[i], state.weightVelocity[i],
                            beta1, beta2, learningRate, epsilon, weightDecay,
                            momentumCorrection, velocityCorrection, true);
                }
            };
        }
        
        // Execute weight updates in parallel
        Parallelization.executeParallel(executor, weightTasks);
        
        // Update biases sequentially (usually small arrays, not worth parallelizing)
        FusedAdamWUpdate.compute(biases, biasGradients, state.biasMomentum, state.biasVelocity,
                               beta1, beta2, learningRate, epsilon, weightDecay,
                               momentumCorrection, velocityCorrection, false);
    }
    
    /**
     * Sequential optimization (fallback when parallelization isn't beneficial).
     */
    private void sequentialOptimize(float[][] weights, float[] biases, float[][] weightGradients, 
                                   float[] biasGradients, AdamWState state, long currentTimeStep) {
        // Pre-compute bias correction factors (avoid redundant computation)
        float momentumCorrection = 1.0f - (float) Math.pow(beta1, currentTimeStep);
        float velocityCorrection = 1.0f - (float) Math.pow(beta2, currentTimeStep);
        
        // Update weights using fused operation
        for (int i = 0; i < weights.length; i++) {
            FusedAdamWUpdate.compute(weights[i], weightGradients[i],
                    state.weightMomentum[i], state.weightVelocity[i],
                    beta1, beta2, learningRate, epsilon, weightDecay,
                    momentumCorrection, velocityCorrection, true);
        }
        
        // Update biases using fused operation (typically no weight decay applied to biases)
        FusedAdamWUpdate.compute(biases, biasGradients, state.biasMomentum, state.biasVelocity,
                               beta1, beta2, learningRate, epsilon, weightDecay,
                               momentumCorrection, velocityCorrection, false);
    }
    
    
    /**
     * State storage for AdamW optimizer per layer.
     * Thread-safe through ConcurrentHashMap access patterns.
     * Simplified now that fused operations handle buffer management internally.
     */
    private static class AdamWState {
        final float[][] weightMomentum;    // m_t for weights
        final float[][] weightVelocity;    // v_t for weights  
        final float[] biasMomentum;        // m_t for biases
        final float[] biasVelocity;        // v_t for biases
        final AtomicLong timeStep = new AtomicLong(0); // t (for bias correction) - atomic for thread safety
        
        AdamWState(float[][] weights, float[] biases) {
            // Initialize momentum and velocity to zeros
            weightMomentum = new float[weights.length][];
            weightVelocity = new float[weights.length][];
            for (int i = 0; i < weights.length; i++) {
                weightMomentum[i] = new float[weights[i].length];
                weightVelocity[i] = new float[weights[i].length];
            }
            
            biasMomentum = new float[biases.length];
            biasVelocity = new float[biases.length];
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
    
    /**
     * @return the weight decay coefficient
     */
    public float getWeightDecay() {
        return weightDecay;
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        out.writeFloat(learningRate);
        out.writeFloat(beta1);
        out.writeFloat(beta2);
        out.writeFloat(epsilon);
        out.writeFloat(weightDecay);
        // Note: We don't serialize the state - it will be rebuilt during training
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize(DataInputStream, int) static method instead");
    }
    
    /**
     * Static method to deserialize an AdamWOptimizer from stream.
     */
    public static AdamWOptimizer deserialize(DataInputStream in, int version) throws IOException {
        float learningRate = in.readFloat();
        float beta1 = in.readFloat();
        float beta2 = in.readFloat();
        float epsilon = in.readFloat();
        float weightDecay = in.readFloat();
        return new AdamWOptimizer(learningRate, beta1, beta2, epsilon, weightDecay);
    }
    
    @Override
    public int getSerializedSize(int version) {
        return 20; // 5 floats
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_ADAMW_OPTIMIZER;
    }
    
    @Override
    public void setLearningRate(float learningRate) {
        if (learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive: " + learningRate);
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
        // The state is created based on the full `allWeights` table.
        AdamWState state = layerStates.computeIfAbsent(stateKey, k -> new AdamWState(allWeights, new float[0]));

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

            // Perform the fused AdamW update on the specific row
            FusedAdamWUpdate.compute(
                    allWeights[weightIndex], gradient,
                    state.weightMomentum[weightIndex], state.weightVelocity[weightIndex],
                    beta1, beta2, learningRate, epsilon, weightDecay,
                    momentumCorrection, velocityCorrection, true); // true = apply weight decay
        }
    }
    
    @Override
    public Optimizer forEmbeddings() {
        // For embeddings: 10x less weight decay, 5x higher learning rate
        // High-cardinality features need higher LR due to sparse updates
        if (weightDecay > 0 || learningRate > 0) {
            float embeddingDecay = weightDecay * 0.1f;  // Less regularization
            float embeddingLR = learningRate * 5.0f;    // Faster learning for sparse features
            return new AdamWOptimizer(embeddingLR, beta1, beta2, epsilon, embeddingDecay);
        }
        return this;
    }
}