package dev.neuronic.net.layers;

import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.common.PooledFloatArray;
import dev.neuronic.net.math.FastRandom;

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;

/**
 * Embedding layer using hash functions to handle unlimited vocabulary sizes.
 * 
 * <p>Instead of maintaining a vocabulary dictionary, this layer:
 * <ol>
 *   <li>Hashes input strings using 3 different MurmurHash3 seeds</li>
 *   <li>Maps each hash to an embedding table position (modulo buckets)</li>
 *   <li>Averages the 3 embeddings for collision resistance</li>
 * </ol>
 * 
 * <p>This trades perfect disambiguation for unlimited scale and zero
 * vocabulary management overhead.
 * 
 * <p><b>IMPORTANT - Weight Decay Behavior:</b>
 * This layer automatically disables weight decay for embeddings, following universal
 * best practices in modern NLP (GPT, BERT, T5, LLaMA). Embedding parameters should
 * not be regularized toward zero as this harms the model's ability to distinguish
 * between hashed features.
 * 
 * <p><b>To re-enable weight decay (rare):</b>
 * <pre>{@code
 * // If you need weight decay on embeddings (not recommended):
 * Optimizer customOptimizer = new AdamWOptimizer(0.001f, 0.01f);
 * NeuralNet net = NeuralNet.newBuilder()
 *     .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
 *     .layer(HashedEmbeddingLayer.spec(hashBuckets, embedDim, customOptimizer))  // Explicit optimizer
 *     .build();
 * }</pre>
 * 
 * <p><b>Why is weight decay disabled?</b>
 * <ul>
 *   <li>Prevents rare features from being regularized to zero</li>
 *   <li>Maintains distinguishability between hashed values</li>
 *   <li>Standard practice in all modern language models</li>
 *   <li>Critical for advertising/RecSys sparse features</li>
 * </ul>
 */
public class HashedEmbeddingLayer implements Layer {
    
    // MurmurHash3 prime seeds for better distribution
    private static final int[] HASH_SEEDS = {
        0x1b873593,  // MurmurHash3 constant c1
        0xcc9e2d51,  // MurmurHash3 constant c2
        0x85ebca6b   // MurmurHash3 mix constant
    };
    
    private final int hashBuckets;
    private final int embeddingDim;
    private final int numHashes;
    private final float[][] embeddings;  // [hashBuckets][embeddingDim]
    private final Optimizer optimizer;
    
    // State management
    private final Map<float[][], State> stateMap = new ConcurrentHashMap<>();
    
    private static class State {
        final float[][] gradientAccum;  // [hashBuckets][embeddingDim]
        final PooledFloatArray bufferPool;
        
        State(int hashBuckets, int embeddingDim) {
            this.gradientAccum = new float[hashBuckets][embeddingDim];
            this.bufferPool = new PooledFloatArray(1024);  // Reasonable default
        }
    }
    
    public HashedEmbeddingLayer(int hashBuckets, int embeddingDim, int numHashes, Optimizer optimizer, FastRandom random) {
        this.hashBuckets = hashBuckets;
        this.embeddingDim = embeddingDim;
        this.numHashes = numHashes;
        this.optimizer = optimizer;
        
        // Initialize embeddings with small random values
        this.embeddings = new float[hashBuckets][embeddingDim];
        float scale = (float) Math.sqrt(2.0 / embeddingDim);
        for (int i = 0; i < hashBuckets; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                embeddings[i][j] = (random.nextFloat() * 2 - 1) * scale;
            }
        }
    }
    
    @Override
    public LayerContext forward(float[] input, boolean isTraining) {
        // Input is expected to be hash codes or string representations
        // For now, we'll handle integer hash codes
        int batchSize = input.length;
        float[] output = new float[batchSize * embeddingDim];
        
        for (int b = 0; b < batchSize; b++) {
            // Get string hash code from input
            int hashCode = (int) input[b];
            
            // Compute embedding positions using multiple hashes
            int[] positions = computeHashPositions(hashCode);
            
            // Average embeddings from all hash positions
            int outputOffset = b * embeddingDim;
            for (int pos : positions) {
                for (int d = 0; d < embeddingDim; d++) {
                    output[outputOffset + d] += embeddings[pos][d];
                }
            }
            
            // Scale by 1/numHashes to get average
            float scale = 1.0f / numHashes;
            for (int d = 0; d < embeddingDim; d++) {
                output[outputOffset + d] *= scale;
            }
        }
        
        // Store input for backward pass
        return new LayerContext(input, null, output);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] gradOutput) {
        LayerContext context = stack[stackIndex];
        float[] input = context.inputs();
        int batchSize = input.length;
        
        // Get or create state
        State state = stateMap.computeIfAbsent(embeddings, k -> new State(hashBuckets, embeddingDim));
        
        // Accumulate gradients for embeddings
        for (int b = 0; b < batchSize; b++) {
            int hashCode = (int) input[b];
            int[] positions = computeHashPositions(hashCode);
            
            int gradOffset = b * embeddingDim;
            float scale = 1.0f / numHashes;
            
            for (int pos : positions) {
                for (int d = 0; d < embeddingDim; d++) {
                    state.gradientAccum[pos][d] += gradOutput[gradOffset + d] * scale;
                }
            }
        }
        
        // Update embeddings using optimizer
        optimizer.optimize(embeddings, new float[0], state.gradientAccum, new float[0]);
        
        // Clear gradient accumulator
        for (int i = 0; i < hashBuckets; i++) {
            Arrays.fill(state.gradientAccum[i], 0f);
        }
        
        // No gradient to propagate back (input layer)
        return null;
    }
    
    /**
     * Compute hash positions using multiple hash functions.
     */
    private int[] computeHashPositions(int hashCode) {
        int[] positions = new int[numHashes];
        
        for (int i = 0; i < numHashes; i++) {
            // MurmurHash3-inspired mixing
            int h = hashCode;
            h ^= h >>> 16;
            h *= HASH_SEEDS[i];
            h ^= h >>> 13;
            h *= 0x5bd1e995;
            h ^= h >>> 15;
            
            // Map to bucket (ensure positive)
            positions[i] = Math.abs(h) % hashBuckets;
        }
        
        return positions;
    }
    
    /**
     * Static method to hash a string to integer for input.
     * This should be called before passing data to the layer.
     */
    public static int hashString(String s) {
        if (s == null) return 0;
        
        // Use Java's string hash as base
        int h = s.hashCode();
        
        // Additional mixing for better distribution
        h ^= h >>> 16;
        h *= 0x85ebca6b;
        h ^= h >>> 13;
        h *= 0xc2b2ae35;
        h ^= h >>> 16;
        
        return h;
    }
    
    @Override
    public int getOutputSize() {
        return embeddingDim;
    }
    
    @Override
    public Optimizer getOptimizer() {
        return optimizer;
    }
    
    public static Spec spec(int hashBuckets, int embeddingDim, Optimizer optimizer) {
        return new Spec(hashBuckets, embeddingDim, 3, optimizer);
    }
    
    /**
     * Layer specification for building hashed embedding layers.
     */
    public static class Spec extends BaseLayerSpec<Spec> {
        private final int hashBuckets;
        private final int numHashes;
        
        public Spec(int hashBuckets, int embeddingDim, int numHashes, Optimizer optimizer) {
            super(embeddingDim, optimizer);
            this.hashBuckets = hashBuckets;
            this.numHashes = numHashes;
        }
        
        @Override
        protected Optimizer getEffectiveOptimizer(Optimizer defaultOptimizer) {
            // First get the base optimizer (with learning rate scaling if needed)
            Optimizer baseOptimizer = super.getEffectiveOptimizer(defaultOptimizer);
            
            // If user didn't explicitly set an optimizer, apply embedding best practices
            if (optimizer == null) {
                // Automatically reduce/disable weight decay for embeddings
                // This follows universal best practices in modern NLP
                return baseOptimizer.forEmbeddings();
            }
            
            // User explicitly set an optimizer - respect their choice
            return baseOptimizer;
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer, FastRandom random) {
            return new HashedEmbeddingLayer(hashBuckets, outputSize, numHashes, effectiveOptimizer, random);
        }
    }
}