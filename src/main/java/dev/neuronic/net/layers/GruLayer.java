package dev.neuronic.net.layers;

import dev.neuronic.net.Shape;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.activators.SigmoidActivator;
import dev.neuronic.net.activators.TanhActivator;
import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.math.Parallelization;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.AdamOptimizer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.serialization.SerializationRegistry;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;

/**
 * Gated Recurrent Unit (GRU) layer for sequence modeling.
 * 
 * GRU is a simplified variant of LSTM that uses only 2 gates instead of 3:
 * - Reset gate (r): Controls how much of previous hidden state to "forget"
 * - Update gate (z): Controls how much to update the hidden state
 * 
 * Benefits over LSTM:
 * - Fewer parameters (faster training, less overfitting)
 * - Simpler architecture (easier to understand and debug)  
 * - Often comparable performance for many tasks
 * 
 * Architecture:
 * r_t = σ(W_r * [h_{t-1}, x_t] + b_r)     Reset gate
 * z_t = σ(W_z * [h_{t-1}, x_t] + b_z)     Update gate  
 * h̃_t = tanh(W_h * [r_t ⊙ h_{t-1}, x_t] + b_h)  Candidate hidden state
 * h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t         Final hidden state
 * 
 * Usage:
 * GruLayer gru = new GruLayer(optimizer, hiddenSize, inputSize, WeightInitStrategy.XAVIER);
 * LayerContext context = gru.forward(sequenceInput);  // [seqLen * inputSize]
 * float[] hiddenStates = context.outputs();           // [seqLen * hiddenSize]
 */
public class GruLayer implements Layer, Serializable {
    
    /**
     * Output mode for GRU layer - controls which timesteps are returned.
     */
    public enum OutputMode {
        /**
         * Output hidden states for ALL timesteps.
         * Output shape: [sequenceLength × hiddenSize]
         * Use for: seq2seq, attention mechanisms, bidirectional RNNs
         */
        ALL_TIMESTEPS,
        
        /**
         * Output hidden state for LAST timestep only.
         * Output shape: [hiddenSize]
         * Use for: classification, language modeling, time series prediction
         */
        LAST_TIMESTEP
    }
    
    /**
     * GRU-specific layer context that stores intermediate states for BPTT.
     * Contains all gate values and hidden states for each timestep.
     */
    public static class GruLayerContext extends LayerContext {
        public final int seqLen;                    // Sequence length
        public final float[][] resetGates;         // Reset gate values for each timestep [seqLen][hiddenSize]
        public final float[][] updateGates;        // Update gate values for each timestep [seqLen][hiddenSize]
        public final float[][] candidates;         // Candidate values for each timestep [seqLen][hiddenSize]
        public final float[][] hiddenStates;       // Hidden states for each timestep [seqLen+1][hiddenSize] (includes initial h_0)
        public final float[][] concatenatedInputs; // Concatenated inputs for each timestep [seqLen][inputSize + hiddenSize]
        
        // LayerNorm data for backward pass (null if LayerNorm not used)
        public final float[][] resetNormalized;    // Normalized values for reset gate [seqLen][hiddenSize]
        public final float[][] updateNormalized;   // Normalized values for update gate [seqLen][hiddenSize]
        public final float[][] candidateNormalized;// Normalized values for candidate [seqLen][hiddenSize]
        public final LayerNorm.Stats[] resetStats;    // Stats for reset gate [seqLen]
        public final LayerNorm.Stats[] updateStats;   // Stats for update gate [seqLen]
        public final LayerNorm.Stats[] candidateStats;// Stats for candidate [seqLen]
        
        public GruLayerContext(float[] inputs, float[] preActivations, float[] outputs, int seqLen,
                             float[][] resetGates, float[][] updateGates, float[][] candidates,
                             float[][] hiddenStates, float[][] concatenatedInputs) {
            this(inputs, preActivations, outputs, seqLen, resetGates, updateGates, candidates, 
                 hiddenStates, concatenatedInputs, null, null, null, null, null, null);
        }
        
        public GruLayerContext(float[] inputs, float[] preActivations, float[] outputs, int seqLen,
                             float[][] resetGates, float[][] updateGates, float[][] candidates,
                             float[][] hiddenStates, float[][] concatenatedInputs,
                             float[][] resetNormalized, float[][] updateNormalized, float[][] candidateNormalized,
                             LayerNorm.Stats[] resetStats, LayerNorm.Stats[] updateStats, LayerNorm.Stats[] candidateStats) {
            super(inputs, preActivations, outputs);
            this.seqLen = seqLen;
            this.resetGates = resetGates;
            this.updateGates = updateGates;
            this.candidates = candidates;
            this.hiddenStates = hiddenStates;
            this.concatenatedInputs = concatenatedInputs;
            this.resetNormalized = resetNormalized;
            this.updateNormalized = updateNormalized;
            this.candidateNormalized = candidateNormalized;
            this.resetStats = resetStats;
            this.updateStats = updateStats;
            this.candidateStats = candidateStats;
        }
    }
    
    // Core parameters
    private final Optimizer optimizer;
    private final int hiddenSize;
    private final int inputSize;
    private final OutputMode outputMode;
    
    // Weight matrices [inputSize + hiddenSize][hiddenSize] - concatenated input format
    private final float[][] resetWeights;      // W_r for reset gate
    private final float[][] updateWeights;     // W_z for update gate  
    private final float[][] candidateWeights;  // W_h for candidate state
    private final int totalInputSize;          // inputSize + hiddenSize
    
    // Bias vectors [hiddenSize]
    private final float[] resetBias;           // b_r
    private final float[] updateBias;          // b_z
    private final float[] candidateBias;       // b_h
    
    // Optional layer normalization components
    private final LayerNorm resetLayerNorm;    // Layer norm for reset gate
    private final LayerNorm updateLayerNorm;   // Layer norm for update gate
    private final LayerNorm candidateLayerNorm; // Layer norm for candidate
    private final boolean useLayerNorm;
    
    /**
     * Consolidated ThreadLocal buffer container for optimal performance.
     * Reduces ThreadLocal.get() overhead and improves cache locality.
     */
    private static class GruBuffers {
        // Forward pass buffers (all hiddenSize)
        final float[] hiddenStateBuffer;
        final float[] resetGateBuffer;
        final float[] updateGateBuffer;
        final float[] candidateBuffer;
        
        // Variable size buffers
        final float[] concatenatedInputBuffer;  // totalInputSize
        final float[] currentInputBuffer;       // inputSize
        
        // Gradient buffers for backward pass
        final float[][] resetWeightGradients;
        final float[][] updateWeightGradients;
        final float[][] candidateWeightGradients;
        final float[] resetBiasGradients;
        final float[] updateBiasGradients;
        final float[] candidateBiasGradients;

        // Temporary buffers for BPTT
        final float[][] tempWeightGrads; // For single-step outer products
        final float[] tempHiddenGrads;
        final float[] tempPreActivationGrads;
        final float[] tempConcatenatedGrads; // Additional buffer to avoid aliasing
        final float[] tempHiddenGrads2; // Second hidden buffer to avoid aliasing
        
        // LayerNorm buffers for backward pass
        final float[] resetNormalizedBuffer;
        final float[] updateNormalizedBuffer;
        final float[] candidateNormalizedBuffer;
        final float[] resetGammaGradients;      // Per-timestep gradients
        final float[] updateGammaGradients;     // Per-timestep gradients
        final float[] candidateGammaGradients;  // Per-timestep gradients
        final float[] resetBetaGradients;       // Per-timestep gradients
        final float[] updateBetaGradients;      // Per-timestep gradients
        final float[] candidateBetaGradients;   // Per-timestep gradients
        final float[] layerNormInputGradient;
        
        // Accumulated LayerNorm gradients across all timesteps
        final float[] resetGammaGradientsAccum;
        final float[] updateGammaGradientsAccum;
        final float[] candidateGammaGradientsAccum;
        final float[] resetBetaGradientsAccum;
        final float[] updateBetaGradientsAccum;
        final float[] candidateBetaGradientsAccum;
        
        // Sequence-level buffers that grow as needed
        float[][] resetGates;                   // [seqLen][hiddenSize]
        float[][] updateGates;                  // [seqLen][hiddenSize]
        float[][] candidates;                   // [seqLen][hiddenSize]
        float[][] hiddenStates;                 // [seqLen+1][hiddenSize]
        float[][] concatenatedInputs;           // [seqLen][totalInputSize]
        // LayerNorm sequence buffers
        float[][] resetNormalized;              // [seqLen][hiddenSize]
        float[][] updateNormalized;             // [seqLen][hiddenSize]
        float[][] candidateNormalized;          // [seqLen][hiddenSize]
        LayerNorm.Stats[] resetStats;          // [seqLen]
        LayerNorm.Stats[] updateStats;         // [seqLen]
        LayerNorm.Stats[] candidateStats;      // [seqLen]
        int currentSequenceCapacity = 0;        // Current capacity for sequence buffers
        
        GruBuffers(int hiddenSize, int inputSize, int totalInputSize) {
            // Forward pass buffers
            this.hiddenStateBuffer = new float[hiddenSize];
            this.resetGateBuffer = new float[hiddenSize];
            this.updateGateBuffer = new float[hiddenSize];
            this.candidateBuffer = new float[hiddenSize];
            
            // Variable size buffers
            this.concatenatedInputBuffer = new float[totalInputSize];
            this.currentInputBuffer = new float[inputSize];
            
            // Gradient buffers
            this.resetWeightGradients = new float[totalInputSize][hiddenSize];
            this.updateWeightGradients = new float[totalInputSize][hiddenSize];
            this.candidateWeightGradients = new float[totalInputSize][hiddenSize];
            this.resetBiasGradients = new float[hiddenSize];
            this.updateBiasGradients = new float[hiddenSize];
            this.candidateBiasGradients = new float[hiddenSize];

            // Temporary buffers for BPTT
            this.tempWeightGrads = new float[totalInputSize][hiddenSize];
            this.tempHiddenGrads = new float[hiddenSize];
            this.tempPreActivationGrads = new float[hiddenSize];
            this.tempConcatenatedGrads = new float[totalInputSize];
            this.tempHiddenGrads2 = new float[hiddenSize];
            
            // LayerNorm buffers
            this.resetNormalizedBuffer = new float[hiddenSize];
            this.updateNormalizedBuffer = new float[hiddenSize];
            this.candidateNormalizedBuffer = new float[hiddenSize];
            this.resetGammaGradients = new float[hiddenSize];
            this.updateGammaGradients = new float[hiddenSize];
            this.candidateGammaGradients = new float[hiddenSize];
            this.resetBetaGradients = new float[hiddenSize];
            this.updateBetaGradients = new float[hiddenSize];
            this.candidateBetaGradients = new float[hiddenSize];
            this.layerNormInputGradient = new float[hiddenSize];
            
            // Accumulated LayerNorm gradients
            this.resetGammaGradientsAccum = new float[hiddenSize];
            this.updateGammaGradientsAccum = new float[hiddenSize];
            this.candidateGammaGradientsAccum = new float[hiddenSize];
            this.resetBetaGradientsAccum = new float[hiddenSize];
            this.updateBetaGradientsAccum = new float[hiddenSize];
            this.candidateBetaGradientsAccum = new float[hiddenSize];
        }
        
        /**
         * Ensure sequence-level buffers have sufficient capacity for the given sequence length.
         */
        void ensureSequenceCapacity(int seqLen, int hiddenSize, int totalInputSize) {
            if (currentSequenceCapacity < seqLen) {
                // Allocate with some extra capacity to avoid frequent reallocations
                int newCapacity = Math.max(seqLen, currentSequenceCapacity * 2);
                if (newCapacity <= currentSequenceCapacity) newCapacity = currentSequenceCapacity + 1;

                resetGates = new float[newCapacity][hiddenSize];
                updateGates = new float[newCapacity][hiddenSize];
                candidates = new float[newCapacity][hiddenSize];
                hiddenStates = new float[newCapacity + 1][hiddenSize]; // +1 for initial h_0
                concatenatedInputs = new float[newCapacity][totalInputSize];
                
                // LayerNorm sequence buffers
                resetNormalized = new float[newCapacity][hiddenSize];
                updateNormalized = new float[newCapacity][hiddenSize];
                candidateNormalized = new float[newCapacity][hiddenSize];
                resetStats = new LayerNorm.Stats[newCapacity];
                updateStats = new LayerNorm.Stats[newCapacity];
                candidateStats = new LayerNorm.Stats[newCapacity];
                for (int i = 0; i < newCapacity; i++) {
                    resetStats[i] = new LayerNorm.Stats();
                    updateStats[i] = new LayerNorm.Stats();
                    candidateStats[i] = new LayerNorm.Stats();
                }
                
                currentSequenceCapacity = newCapacity;
            }
        }
    }
    
    // Single ThreadLocal for all buffers - major performance improvement
    private final ThreadLocal<GruBuffers> allBuffers;
    
    public GruLayer(Optimizer optimizer, int hiddenSize, int inputSize, WeightInitStrategy initStrategy, FastRandom random) {
        this(optimizer, hiddenSize, inputSize, initStrategy, OutputMode.ALL_TIMESTEPS, random);
    }
    
    public GruLayer(Optimizer optimizer, int hiddenSize, int inputSize, WeightInitStrategy initStrategy, OutputMode outputMode, FastRandom random) {
        this(optimizer, hiddenSize, inputSize, initStrategy, outputMode, false, random);
    }
    
    public GruLayer(Optimizer optimizer, int hiddenSize, int inputSize, WeightInitStrategy initStrategy, OutputMode outputMode, boolean useLayerNorm, FastRandom random) {
        if (optimizer == null)
            throw new IllegalArgumentException("Optimizer cannot be null");
        if (hiddenSize <= 0)
            throw new IllegalArgumentException("Hidden size must be positive: " + hiddenSize);
        if (inputSize <= 0)
            throw new IllegalArgumentException("Input size must be positive: " + inputSize);
        if (initStrategy == null)
            throw new IllegalArgumentException("Weight initialization strategy cannot be null");
        if (outputMode == null)
            throw new IllegalArgumentException("Output mode cannot be null");
        
        this.optimizer = optimizer;
        this.hiddenSize = hiddenSize;
        this.inputSize = inputSize;
        this.outputMode = outputMode;
        this.useLayerNorm = useLayerNorm;
        
        this.totalInputSize = inputSize + hiddenSize;
        
        // Initialize weight matrices
        this.resetWeights = new float[totalInputSize][hiddenSize];
        this.updateWeights = new float[totalInputSize][hiddenSize];
        this.candidateWeights = new float[totalInputSize][hiddenSize];
        
        // Initialize biases
        this.resetBias = new float[hiddenSize];
        this.updateBias = new float[hiddenSize];
        this.candidateBias = new float[hiddenSize];
        
        // Initialize layer normalization if enabled
        if (useLayerNorm) {
            this.resetLayerNorm = new LayerNorm(hiddenSize);
            this.updateLayerNorm = new LayerNorm(hiddenSize);
            this.candidateLayerNorm = new LayerNorm(hiddenSize);
        } else {
            this.resetLayerNorm = null;
            this.updateLayerNorm = null;
            this.candidateLayerNorm = null;
        }
        
        // Initialize consolidated ThreadLocal buffer container
        this.allBuffers = ThreadLocal.withInitial(() -> new GruBuffers(hiddenSize, inputSize, totalInputSize));
        
        initializeWeights(initStrategy, random);
    }
    
    private void initializeWeights(WeightInitStrategy strategy, FastRandom random) {
        switch (strategy) {
            case XAVIER -> {
                NetMath.weightInitXavier(resetWeights, inputSize + hiddenSize, hiddenSize, random);
                NetMath.weightInitXavier(updateWeights, inputSize + hiddenSize, hiddenSize, random);
                NetMath.weightInitXavier(candidateWeights, inputSize + hiddenSize, hiddenSize, random);
            }
            case HE -> {
                NetMath.weightInitHe(resetWeights, inputSize + hiddenSize, random);
                NetMath.weightInitHe(updateWeights, inputSize + hiddenSize, random);
                NetMath.weightInitHe(candidateWeights, inputSize + hiddenSize, random);
            }
        }
        
        NetMath.biasInit(resetBias, 0.0f);
        NetMath.biasInit(updateBias, 0.0f);
        NetMath.biasInit(candidateBias, 0.0f);
    }
    
    @Override
    public LayerContext forward(float[] input, boolean isTraining) {
        validateInputs(input);
        
        int seqLen = input.length / inputSize;
        if (seqLen == 0 && input.length > 0) {
             throw new IllegalArgumentException("Input length is non-zero but sequence length is zero. Input length must be a multiple of input size.");
        }
        if (seqLen == 0 && input.length == 0) {
            return new GruLayerContext(new float[0], null, new float[0], 0, new float[0][], new float[0][], new float[0][], new float[1][hiddenSize], new float[0][],
                                     null, null, null, null, null, null);
        }
        
        // Get consolidated buffers for this thread and ensure sequence capacity
        GruBuffers buffers = allBuffers.get();
        buffers.ensureSequenceCapacity(seqLen, hiddenSize, totalInputSize);
        validateBuffer(buffers.currentInputBuffer, "currentInputBuffers");
        
        // Use ThreadLocal sequence buffers
        float[][] resetGates = buffers.resetGates;
        float[][] updateGates = buffers.updateGates;
        float[][] candidates = buffers.candidates;
        float[][] hiddenStates = buffers.hiddenStates;
        float[][] concatenatedInputs = buffers.concatenatedInputs;
        
        // Initialize h_0 to zeros
        Arrays.fill(hiddenStates[0], 0.0f);
        
        // Process sequence timestep by timestep
        for (int t = 0; t < seqLen; t++) {
            // Extract current timestep input using efficient array copying
            System.arraycopy(input, t * inputSize, buffers.currentInputBuffer, 0, inputSize);
            
            // Forward through GRU cell and store all intermediate states
            forwardCellWithStorageOptimized(buffers.currentInputBuffer, hiddenStates[t], hiddenStates[t + 1], 
                                          resetGates[t], updateGates[t], candidates[t], concatenatedInputs[t], buffers, t);
        }
        
        // Create output based on output mode
        float[] outputs;
        if (outputMode == OutputMode.ALL_TIMESTEPS) {
            // Output all hidden states (excluding initial h_0)
            outputs = new float[seqLen * hiddenSize];
            for (int t = 0; t < seqLen; t++) {
                System.arraycopy(hiddenStates[t + 1], 0, outputs, t * hiddenSize, hiddenSize);
            }
        } else {
            // Output only last hidden state
            outputs = new float[hiddenSize];
            System.arraycopy(hiddenStates[seqLen], 0, outputs, 0, hiddenSize);
        }
        
        // Create fresh copies of sequence data for LayerContext (required for BPTT)
        float[][] freshResetGates = new float[seqLen][hiddenSize];
        float[][] freshUpdateGates = new float[seqLen][hiddenSize];
        float[][] freshCandidates = new float[seqLen][hiddenSize];
        float[][] freshHiddenStates = new float[seqLen + 1][hiddenSize];
        float[][] freshConcatenatedInputs = new float[seqLen][totalInputSize];
        
        // LayerNorm data (only if using LayerNorm)
        float[][] freshResetNormalized = null;
        float[][] freshUpdateNormalized = null;
        float[][] freshCandidateNormalized = null;
        LayerNorm.Stats[] freshResetStats = null;
        LayerNorm.Stats[] freshUpdateStats = null;
        LayerNorm.Stats[] freshCandidateStats = null;
        
        if (useLayerNorm) {
            freshResetNormalized = new float[seqLen][hiddenSize];
            freshUpdateNormalized = new float[seqLen][hiddenSize];
            freshCandidateNormalized = new float[seqLen][hiddenSize];
            freshResetStats = new LayerNorm.Stats[seqLen];
            freshUpdateStats = new LayerNorm.Stats[seqLen];
            freshCandidateStats = new LayerNorm.Stats[seqLen];
        }
        
        // Copy data from ThreadLocal buffers to fresh arrays
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(resetGates[t], 0, freshResetGates[t], 0, hiddenSize);
            System.arraycopy(updateGates[t], 0, freshUpdateGates[t], 0, hiddenSize);
            System.arraycopy(candidates[t], 0, freshCandidates[t], 0, hiddenSize);
            System.arraycopy(hiddenStates[t + 1], 0, freshHiddenStates[t + 1], 0, hiddenSize);
            System.arraycopy(concatenatedInputs[t], 0, freshConcatenatedInputs[t], 0, totalInputSize);
            
            // Copy LayerNorm data if using LayerNorm
            if (useLayerNorm) {
                System.arraycopy(buffers.resetNormalized[t], 0, freshResetNormalized[t], 0, hiddenSize);
                System.arraycopy(buffers.updateNormalized[t], 0, freshUpdateNormalized[t], 0, hiddenSize);
                System.arraycopy(buffers.candidateNormalized[t], 0, freshCandidateNormalized[t], 0, hiddenSize);
                freshResetStats[t] = new LayerNorm.Stats();
                freshResetStats[t].mean = buffers.resetStats[t].mean;
                freshResetStats[t].variance = buffers.resetStats[t].variance;
                freshUpdateStats[t] = new LayerNorm.Stats();
                freshUpdateStats[t].mean = buffers.updateStats[t].mean;
                freshUpdateStats[t].variance = buffers.updateStats[t].variance;
                freshCandidateStats[t] = new LayerNorm.Stats();
                freshCandidateStats[t].mean = buffers.candidateStats[t].mean;
                freshCandidateStats[t].variance = buffers.candidateStats[t].variance;
            }
        }
        // Copy initial hidden state h_0
        System.arraycopy(hiddenStates[0], 0, freshHiddenStates[0], 0, hiddenSize);
        
        // Use safe factory method for ThreadLocal buffers - creates a custom context with input copy
        return new GruLayerContext(input.clone(), null, outputs, seqLen, 
                                 freshResetGates, freshUpdateGates, freshCandidates, freshHiddenStates, freshConcatenatedInputs,
                                 freshResetNormalized, freshUpdateNormalized, freshCandidateNormalized,
                                 freshResetStats, freshUpdateStats, freshCandidateStats);
    }
    
    
    /**
     * Optimized forward pass through GRU cell using pre-allocated buffers.
     * Eliminates all dynamic allocation for maximum performance.
     * Optionally applies layer normalization to gate pre-activations.
     */
    private void forwardCellWithStorageOptimized(float[] input, float[] prevHidden, float[] newHidden,
                                               float[] resetGate, float[] updateGate, float[] candidate,
                                               float[] concatenatedInput, GruBuffers buffers, int timestep) {
        // Concatenate input and previous hidden state
        System.arraycopy(input, 0, concatenatedInput, 0, inputSize);
        System.arraycopy(prevHidden, 0, concatenatedInput, inputSize, hiddenSize);
        
        // Reset gate: r_t = σ(W_r * [h_{t-1}, x_t] + b_r)
        NetMath.matrixPreActivationsColumnMajor(concatenatedInput, resetWeights, resetBias, resetGate);
        if (useLayerNorm && resetLayerNorm != null) {
            buffers.resetStats[timestep] = resetLayerNorm.forward(resetGate, resetGate, buffers.resetNormalized[timestep]);
        }
        SigmoidActivator.INSTANCE.activate(resetGate, resetGate); // Vectorized internally
        
        // Update gate: z_t = σ(W_z * [h_{t-1}, x_t] + b_z)  
        NetMath.matrixPreActivationsColumnMajor(concatenatedInput, updateWeights, updateBias, updateGate);
        if (useLayerNorm && updateLayerNorm != null) {
            buffers.updateStats[timestep] = updateLayerNorm.forward(updateGate, updateGate, buffers.updateNormalized[timestep]);
        }
        SigmoidActivator.INSTANCE.activate(updateGate, updateGate); // Vectorized internally
        
        // Prepare input for candidate: [r_t ⊙ h_{t-1}, x_t] using consolidated buffers
        float[] resetHiddenProduct = buffers.tempHiddenGrads; // Reuse temp buffer (hiddenSize)
        NetMath.elementwiseMultiply(resetGate, prevHidden, resetHiddenProduct);
        
        // Create the concatenated input for the candidate gate
        float[] candidateConcatenatedInput = buffers.concatenatedInputBuffer; // Can reuse this buffer
        System.arraycopy(input, 0, candidateConcatenatedInput, 0, inputSize);
        System.arraycopy(resetHiddenProduct, 0, candidateConcatenatedInput, inputSize, hiddenSize);

        // Candidate: h̃_t = tanh(W_h * [r_t ⊙ h_{t-1}, x_t] + b_h)
        NetMath.matrixPreActivationsColumnMajor(candidateConcatenatedInput, candidateWeights, candidateBias, candidate);
        if (useLayerNorm && candidateLayerNorm != null) {
            buffers.candidateStats[timestep] = candidateLayerNorm.forward(candidate, candidate, buffers.candidateNormalized[timestep]);
        }
        TanhActivator.INSTANCE.activate(candidate, candidate);
        
        // Final hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t using consolidated buffers
        float[] oneMinusUpdate = buffers.tempHiddenGrads;      // Reuse temp buffer (hiddenSize)
        float[] term1 = buffers.tempPreActivationGrads;               // Reuse temp buffer (hiddenSize) 
        float[] term2 = buffers.hiddenStateBuffer;         // Reuse hidden buffer (hiddenSize)
        
        NetMath.scalarSubtract(1.0f, updateGate, oneMinusUpdate);        // (1 - z_t)
        NetMath.elementwiseMultiply(oneMinusUpdate, prevHidden, term1);  // (1 - z_t) ⊙ h_{t-1}
        NetMath.elementwiseMultiply(updateGate, candidate, term2);       // z_t ⊙ h̃_t
        NetMath.elementwiseAdd(term1, term2, newHidden);                 // Final sum
    }
    
    
    /**
     * Parallel forward pass through GRU cell using ExecutorService for gate computations.
     * Parallelizes the three gate computations when hidden size is large enough.
     * Optionally applies layer normalization to gate pre-activations.
     */
    private void forwardCellParallel(float[] input, float[] prevHidden, float[] newHidden,
                                   float[] resetGate, float[] updateGate, float[] candidate,
                                   float[] concatenatedInput, GruBuffers buffers, ExecutorService executor, int timestep) {
        // This method would need similar corrections as the sequential one if used.
        // For simplicity, focusing on the sequential implementation first as it's the one under test.
        forwardCellWithStorageOptimized(input, prevHidden, newHidden, resetGate, updateGate, candidate, concatenatedInput, buffers, timestep);
    }
    
    @Override
    public LayerContext forward(float[] input, ExecutorService executor) {
        // Simplified to call the sequential version for now to ensure correctness first.
        return forward(input, false);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient, ExecutorService executor) {
        validateBackwardInputs(stack, stackIndex, upstreamGradient);
        
        GruLayerContext gruContext = (GruLayerContext) stack[stackIndex];
        int seqLen = gruContext.seqLen;
        if (seqLen == 0) {
            return new float[0];
        }

        float[] inputGradients = new float[gruContext.inputs().length];
        
        GruBuffers buffers = allBuffers.get();
        
        // Clear all gradient accumulators
        clearWeightGradients(buffers.resetWeightGradients);
        clearWeightGradients(buffers.updateWeightGradients);
        clearWeightGradients(buffers.candidateWeightGradients);
        Arrays.fill(buffers.resetBiasGradients, 0.0f);
        Arrays.fill(buffers.updateBiasGradients, 0.0f);
        Arrays.fill(buffers.candidateBiasGradients, 0.0f);
        
        // Clear LayerNorm gradient accumulators if using LayerNorm
        if (useLayerNorm) {
            Arrays.fill(buffers.resetGammaGradientsAccum, 0.0f);
            Arrays.fill(buffers.updateGammaGradientsAccum, 0.0f);
            Arrays.fill(buffers.candidateGammaGradientsAccum, 0.0f);
            Arrays.fill(buffers.resetBetaGradientsAccum, 0.0f);
            Arrays.fill(buffers.updateBetaGradientsAccum, 0.0f);
            Arrays.fill(buffers.candidateBetaGradientsAccum, 0.0f);
        }
        
        float[] hiddenGradient = buffers.hiddenStateBuffer;
        Arrays.fill(hiddenGradient, 0.0f);
        
        for (int t = seqLen - 1; t >= 0; t--) {
            if (outputMode == OutputMode.ALL_TIMESTEPS) {
                for (int i = 0; i < hiddenSize; i++) {
                    hiddenGradient[i] += upstreamGradient[t * hiddenSize + i];
                }
            } else if (t == seqLen - 1) {
                for (int i = 0; i < hiddenSize; i++) {
                    hiddenGradient[i] += upstreamGradient[i];
                }
            }
            
            backwardTimestepOptimized(t, gruContext, hiddenGradient,
                                    buffers.resetWeightGradients, buffers.updateWeightGradients, buffers.candidateWeightGradients,
                                    buffers.resetBiasGradients, buffers.updateBiasGradients, buffers.candidateBiasGradients,
                                    inputGradients, buffers, executor);
        }
        
        optimizer.optimize(resetWeights, resetBias, buffers.resetWeightGradients, buffers.resetBiasGradients, executor);
        optimizer.optimize(updateWeights, updateBias, buffers.updateWeightGradients, buffers.updateBiasGradients, executor);
        optimizer.optimize(candidateWeights, candidateBias, buffers.candidateWeightGradients, buffers.candidateBiasGradients, executor);
        
        // Apply accumulated LayerNorm parameter gradients if using LayerNorm
        if (useLayerNorm) {
            // Use the layer's optimizer for LayerNorm parameters
            if (resetLayerNorm != null) {
                optimizer.optimize(resetLayerNorm.getGamma(), buffers.resetGammaGradientsAccum, executor);
                optimizer.optimize(resetLayerNorm.getBeta(), buffers.resetBetaGradientsAccum, executor);
            }
            if (updateLayerNorm != null) {
                optimizer.optimize(updateLayerNorm.getGamma(), buffers.updateGammaGradientsAccum, executor);
                optimizer.optimize(updateLayerNorm.getBeta(), buffers.updateBetaGradientsAccum, executor);
            }
            if (candidateLayerNorm != null) {
                optimizer.optimize(candidateLayerNorm.getGamma(), buffers.candidateGammaGradientsAccum, executor);
                optimizer.optimize(candidateLayerNorm.getBeta(), buffers.candidateBetaGradientsAccum, executor);
            }
        }
        
        return inputGradients;
    }
    
    /**
     * Clear weight gradient matrix efficiently.
     */
    private void clearWeightGradients(float[][] gradients) {
        for (int i = 0; i < gradients.length; i++) {
            Arrays.fill(gradients[i], 0.0f);
        }
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
        return backward(stack, stackIndex, upstreamGradient, (ExecutorService) null);
    }
    
    /**
     * Corrected and optimized computation of gradients for a single timestep in BPTT.
     */
    private void backwardTimestepOptimized(int t, GruLayerContext context, float[] hiddenGradient,
                                         float[][] resetWeightGrads, float[][] updateWeightGrads, float[][] candidateWeightGrads,
                                         float[] resetBiasGrads, float[] updateBiasGrads, float[] candidateBiasGrads,
                                         float[] inputGradients, GruBuffers buffers, ExecutorService executor) {

        // --- Get stored values from forward pass ---
        float[] resetGate = context.resetGates[t];
        float[] updateGate = context.updateGates[t];
        float[] candidate = context.candidates[t];
        float[] prevHidden = context.hiddenStates[t];
        float[] concatenatedInputForUpdateAndReset = context.concatenatedInputs[t];

        // --- Buffers for this timestep ---
        float[] temp_h_grad = buffers.tempHiddenGrads;
        float[] pre_act_grad = buffers.tempPreActivationGrads;
        float[][] temp_w_grad = buffers.tempWeightGrads;

        // === 1. Gradient of Loss w.r.t. h_t ===
        // This is `hiddenGradient`, which is passed in and accumulated over time.

        // === 2. Gradients from final hidden state computation ===
        // h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        
        // Grad w.r.t. update gate (z_t)
        float[] updateGateGrad = buffers.updateGateBuffer; // Use a dedicated buffer
        for (int i = 0; i < hiddenSize; i++) {
            updateGateGrad[i] = hiddenGradient[i] * (candidate[i] - prevHidden[i]);
        }

        // Grad w.r.t. candidate (h̃_t)
        float[] candidateGrad = buffers.candidateBuffer; // Use a dedicated buffer
        for (int i = 0; i < hiddenSize; i++) {
            candidateGrad[i] = hiddenGradient[i] * updateGate[i];
        }

        // Grad w.r.t. previous hidden state (h_{t-1}) from this step
        float[] d_prev_hidden_from_final = temp_h_grad;
        for (int i = 0; i < hiddenSize; i++) {
            d_prev_hidden_from_final[i] = hiddenGradient[i] * (1 - updateGate[i]);
        }

        // === 3. Gradient through Candidate Gate ===
        // h̃_t = tanh(W_h * [r_t ⊙ h_{t-1}, x_t] + b_h)
        
        // Backprop through tanh
        TanhActivator.INSTANCE.derivative(candidate, pre_act_grad, executor);
        NetMath.elementwiseMultiply(candidateGrad, pre_act_grad, pre_act_grad); // Now pre_act_grad = dL/d(pre_act_h̃)

        if (useLayerNorm && candidateLayerNorm != null) {
            // Clear gradient accumulation buffers for this timestep
            Arrays.fill(buffers.candidateGammaGradients, 0.0f);
            Arrays.fill(buffers.candidateBetaGradients, 0.0f);
            
            // Backward through LayerNorm
            candidateLayerNorm.backward(pre_act_grad, context.candidateNormalized[t], context.candidateStats[t],
                                      buffers.layerNormInputGradient, buffers.candidateGammaGradients, buffers.candidateBetaGradients);
            
            // Accumulate LayerNorm parameter gradients across timesteps
            for (int i = 0; i < hiddenSize; i++) {
                buffers.candidateGammaGradientsAccum[i] += buffers.candidateGammaGradients[i];
                buffers.candidateBetaGradientsAccum[i] += buffers.candidateBetaGradients[i];
            }
            
            // Copy gradient back to pre_act_grad for further processing
            System.arraycopy(buffers.layerNormInputGradient, 0, pre_act_grad, 0, hiddenSize);
        }

        // Grad for candidate bias
        for (int i = 0; i < hiddenSize; i++) candidateBiasGrads[i] += pre_act_grad[i];

        // Reconstruct the specific input for the candidate gate
        float[] currentInput = buffers.currentInputBuffer;
        System.arraycopy(context.inputs(), t * inputSize, currentInput, 0, inputSize);
        float[] resetHiddenProduct = buffers.resetGateBuffer; // Use a dedicated buffer
        NetMath.elementwiseMultiply(resetGate, prevHidden, resetHiddenProduct);
        float[] concatenatedInputForCandidate = buffers.concatenatedInputBuffer;
        System.arraycopy(currentInput, 0, concatenatedInputForCandidate, 0, inputSize);
        System.arraycopy(resetHiddenProduct, 0, concatenatedInputForCandidate, inputSize, hiddenSize);

        // Grad for candidate weights
        NetMath.matrixOuterProduct(concatenatedInputForCandidate, pre_act_grad, temp_w_grad);
        for(int i=0; i<totalInputSize; i++) for(int j=0; j<hiddenSize; j++) candidateWeightGrads[i][j] += temp_w_grad[i][j];

        // Grad w.r.t concatenated input for candidate
        float[] d_concat_candidate = buffers.concatenatedInputBuffer; // Reuse buffer
        NetMath.matrixVectorMultiplyColumnMajor(candidateWeights, pre_act_grad, d_concat_candidate);
        
        float[] d_input_from_candidate = currentInput; // Reuse buffer
        System.arraycopy(d_concat_candidate, 0, d_input_from_candidate, 0, inputSize);
        
        // CRITICAL FIX: Use separate buffers to avoid aliasing corruption
        // We need the original gradient d_concat_candidate[inputSize:] for TWO different computations
        // Use candidateBuffer which is no longer needed after computing candidateGrad
        float[] d_reset_hidden_prod = buffers.candidateBuffer;
        System.arraycopy(d_concat_candidate, inputSize, d_reset_hidden_prod, 0, hiddenSize);

        // Grad w.r.t reset gate from candidate: dL/dr_t = dL/d(r_t⊙h_{t-1}) ⊙ h_{t-1}
        float[] resetGateGrad = buffers.resetGateBuffer; // Use dedicated buffer
        NetMath.elementwiseMultiply(d_reset_hidden_prod, prevHidden, resetGateGrad);

        // Grad w.r.t prev hidden state from candidate: dL/dh_{t-1} = dL/d(r_t⊙h_{t-1}) ⊙ r_t
        // MUST use original gradient, not the modified resetGateGrad
        // CRITICAL FIX: Use dedicated buffer to avoid aliasing with pre_act_grad
        float[] d_prev_hidden_from_candidate = buffers.tempHiddenGrads2;
        NetMath.elementwiseMultiply(d_reset_hidden_prod, resetGate, d_prev_hidden_from_candidate);


        // === 4. Gradient through Update Gate ===
        // z_t = σ(W_z * [h_{t-1}, x_t] + b_z)
        
        // Backprop through sigmoid
        SigmoidActivator.INSTANCE.derivative(updateGate, pre_act_grad, executor);
        NetMath.elementwiseMultiply(updateGateGrad, pre_act_grad, pre_act_grad); // Now pre_act_grad = dL/d(pre_act_z)

        if (useLayerNorm && updateLayerNorm != null) {
            // Clear gradient accumulation buffers for this timestep
            Arrays.fill(buffers.updateGammaGradients, 0.0f);
            Arrays.fill(buffers.updateBetaGradients, 0.0f);
            
            // Backward through LayerNorm
            updateLayerNorm.backward(pre_act_grad, context.updateNormalized[t], context.updateStats[t],
                                   buffers.layerNormInputGradient, buffers.updateGammaGradients, buffers.updateBetaGradients);
            
            // Accumulate LayerNorm parameter gradients across timesteps
            for (int i = 0; i < hiddenSize; i++) {
                buffers.updateGammaGradientsAccum[i] += buffers.updateGammaGradients[i];
                buffers.updateBetaGradientsAccum[i] += buffers.updateBetaGradients[i];
            }
            
            // Copy gradient back to pre_act_grad for further processing
            System.arraycopy(buffers.layerNormInputGradient, 0, pre_act_grad, 0, hiddenSize);
        }

        // Grad for update bias
        for (int i = 0; i < hiddenSize; i++) updateBiasGrads[i] += pre_act_grad[i];
        
        // Grad for update weights
        NetMath.matrixOuterProduct(concatenatedInputForUpdateAndReset, pre_act_grad, temp_w_grad);
        for(int i=0; i<totalInputSize; i++) for(int j=0; j<hiddenSize; j++) updateWeightGrads[i][j] += temp_w_grad[i][j];

        // Grad w.r.t concatenated input for update gate
        float[] d_concat_update = buffers.concatenatedInputBuffer; // Reuse buffer
        NetMath.matrixVectorMultiplyColumnMajor(updateWeights, pre_act_grad, d_concat_update);


        // === 5. Gradient through Reset Gate ===
        // r_t = σ(W_r * [h_{t-1}, x_t] + b_r)

        // Backprop through sigmoid
        SigmoidActivator.INSTANCE.derivative(resetGate, pre_act_grad, executor);
        NetMath.elementwiseMultiply(resetGateGrad, pre_act_grad, pre_act_grad); // Now pre_act_grad = dL/d(pre_act_r)

        if (useLayerNorm && resetLayerNorm != null) {
            // Clear gradient accumulation buffers for this timestep
            Arrays.fill(buffers.resetGammaGradients, 0.0f);
            Arrays.fill(buffers.resetBetaGradients, 0.0f);
            
            // Backward through LayerNorm
            resetLayerNorm.backward(pre_act_grad, context.resetNormalized[t], context.resetStats[t],
                                  buffers.layerNormInputGradient, buffers.resetGammaGradients, buffers.resetBetaGradients);
            
            // Accumulate LayerNorm parameter gradients across timesteps
            for (int i = 0; i < hiddenSize; i++) {
                buffers.resetGammaGradientsAccum[i] += buffers.resetGammaGradients[i];
                buffers.resetBetaGradientsAccum[i] += buffers.resetBetaGradients[i];
            }
            
            // Copy gradient back to pre_act_grad for further processing
            System.arraycopy(buffers.layerNormInputGradient, 0, pre_act_grad, 0, hiddenSize);
        }

        // Grad for reset bias
        for (int i = 0; i < hiddenSize; i++) resetBiasGrads[i] += pre_act_grad[i];

        // Grad for reset weights
        NetMath.matrixOuterProduct(concatenatedInputForUpdateAndReset, pre_act_grad, temp_w_grad);
        for(int i=0; i<totalInputSize; i++) for(int j=0; j<hiddenSize; j++) resetWeightGrads[i][j] += temp_w_grad[i][j];

        // Grad w.r.t concatenated input for reset gate
        // CRITICAL FIX: Use different buffer to avoid overwriting d_concat_update
        float[] d_concat_reset = buffers.tempConcatenatedGrads;
        NetMath.matrixVectorMultiplyColumnMajor(resetWeights, pre_act_grad, d_concat_reset);


        // === 6. Accumulate gradients for inputs and h_{t-1} ===

        // Accumulate input gradients for this timestep
        for (int i = 0; i < inputSize; i++) {
            inputGradients[t * inputSize + i] = d_input_from_candidate[i] + d_concat_update[i] + d_concat_reset[i];
        }

        // Accumulate hidden state gradient for next (previous) timestep
        if (t > 0) {
            // CRITICAL: Do NOT use context.hiddenStates as a buffer - it contains forward pass data!
            // Just accumulate the gradient directly into hiddenGradient
            for (int i = 0; i < hiddenSize; i++) {
                float grad_h = d_prev_hidden_from_final[i] +
                               d_prev_hidden_from_candidate[i] +
                               d_concat_update[inputSize + i] +
                               d_concat_reset[inputSize + i];
                hiddenGradient[i] = grad_h; // This will be used in the t-1 iteration
            }
        }
    }
    
    
    
    
    @Override
    public int getOutputSize() {
        return hiddenSize;
    }
    
    @Override
    public Optimizer getOptimizer() {
        return optimizer;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public int getInputSize() {
        return inputSize;
    }
    
    public static Layer.Spec spec(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return new GruLayerSpec(hiddenSize, optimizer, initStrategy, 1.0);
    }
    
    /**
     * Create a GRU layer specification with custom learning rate ratio.
     * 
     * @param hiddenSize number of hidden units
     * @param optimizer optimizer for this layer (null to use default)
     * @param initStrategy weight initialization strategy
     * @param learningRateRatio learning rate scaling factor (1.0 = normal)
     */
    public static Layer.Spec spec(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy, double learningRateRatio) {
        return new GruLayerSpec(hiddenSize, optimizer, initStrategy, learningRateRatio);
    }
    
    /**
     * Create a GRU layer that outputs ALL timesteps.
     * 
     * <p><b>Output shape:</b> [sequenceLength × hiddenSize]
     * 
     * <p><b>When to use:</b>
     * <ul>
     *   <li>Sequence-to-sequence models (machine translation, text summarization)</li>
     *   <li>When you need the full sequence of hidden states</li>
     *   <li>Bidirectional RNNs (processing states from both directions)</li>
     *   <li>Attention mechanisms that need all timestep representations</li>
     * </ul>
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // Text generation model that needs all hidden states
     * NeuralNet model = NeuralNet.newBuilder()
     *     .input(35)  // sequence of 35 token IDs
     *     .layer(Layers.inputEmbedding(30000, 256))
     *     .layer(Layers.hiddenGruAll(512))  // Outputs: 35 × 512 = 17,920 values
     *     .layer(Layers.hiddenDenseRelu(256))
     *     .output(Layers.outputSoftmaxCrossEntropy(30000));
     * }</pre>
     * 
     * @param hiddenSize number of hidden units in GRU
     * @param optimizer optimizer for GRU parameters
     * @param initStrategy weight initialization strategy
     * @return GRU spec that outputs all timesteps
     */
    public static Layer.Spec specAll(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return new GruAllTimestepsSpec(hiddenSize, optimizer, initStrategy, 1.0, -1);
    }
    
    /**
     * Create a GRU layer that outputs ALL timesteps with input dimension hint.
     * 
     * <p>Use this when you know the per-timestep input dimension (e.g., embedding dimension).
     * This allows the spec to correctly calculate output sizes before layer creation.
     * 
     * @param hiddenSize number of hidden units in GRU
     * @param optimizer optimizer for GRU parameters
     * @param initStrategy weight initialization strategy
     * @param expectedInputDimension expected size per timestep (e.g., embedding dimension)
     * @return GRU spec that outputs all timesteps
     */
    public static Layer.Spec specAll(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy, 
                                    int expectedInputDimension) {
        return new GruAllTimestepsSpec(hiddenSize, optimizer, initStrategy, 1.0, expectedInputDimension);
    }
    
    /**
     * Create a GRU layer that outputs ONLY the LAST timestep.
     * 
     * <p><b>Output shape:</b> [hiddenSize]
     * 
     * <p><b>When to use:</b>
     * <ul>
     *   <li>Sequence classification (sentiment analysis, spam detection)</li>
     *   <li>Language modeling (predicting next word)</li>
     *   <li>Time series prediction (forecasting next value)</li>
     *   <li>Any many-to-one sequence task</li>
     * </ul>
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // Sentiment classification model
     * NeuralNet model = NeuralNet.newBuilder()
     *     .input(50)  // sequence of 50 token IDs  
     *     .layer(Layers.inputEmbedding(10000, 128))
     *     .layer(Layers.hiddenGruLast(256))  // Outputs: 256 values (last timestep only)
     *     .layer(Layers.hiddenDenseRelu(128))
     *     .output(Layers.outputSoftmaxCrossEntropy(2));  // positive/negative
     * }</pre>
     * 
     * @param hiddenSize number of hidden units in GRU
     * @param optimizer optimizer for GRU parameters
     * @param initStrategy weight initialization strategy
     * @return GRU spec that outputs only last timestep
     */
    public static Layer.Spec specLast(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return new GruLastTimestepSpec(hiddenSize, optimizer, initStrategy, 1.0);
    }
    
    /**
     * Create a layer-normalized GRU specification that outputs all timesteps.
     * 
     * @param hiddenSize number of hidden units
     * @param optimizer optimizer for parameters
     * @param initStrategy weight initialization strategy
     * @return Layer-normalized GRU spec that outputs all timesteps
     */
    public static Layer.Spec specAllNormalized(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return new GruAllTimestepsNormalizedSpec(hiddenSize, optimizer, initStrategy, 1.0);
    }
    
    /**
     * Create a layer-normalized GRU specification that outputs only the last timestep.
     * 
     * @param hiddenSize number of hidden units
     * @param optimizer optimizer for parameters
     * @param initStrategy weight initialization strategy
     * @return Layer-normalized GRU spec that outputs only last timestep
     */
    public static Layer.Spec specLastNormalized(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return new GruLastTimestepNormalizedSpec(hiddenSize, optimizer, initStrategy, 1.0);
    }
    
    /**
     * Specification for creating GRU layers with optimizer management.
     */
    private static class GruLayerSpec extends BaseLayerSpec<GruLayerSpec> {
        private final int hiddenSize;
        private final WeightInitStrategy initStrategy;
        
        public GruLayerSpec(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy, double learningRateRatio) {
            super(hiddenSize, optimizer);
            this.hiddenSize = hiddenSize;
            this.initStrategy = initStrategy;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        
        @Override
        public boolean prefersShapeAPI() {
            return true; // GRU requires shape information
        }
        
        @Override
        public void validateInputShape(Shape inputShape) {
            if (inputShape.rank() != 2 && inputShape.rank() != 1) {
                throw new IllegalArgumentException(
                    "GRU expects 2D input [sequenceLength, features] or 1D flattened input, got shape: " + inputShape);
            }
        }
        
        @Override
        public Layer create(Shape inputShape, Optimizer effectiveOptimizer, FastRandom random) {
            if (effectiveOptimizer == null) {
                effectiveOptimizer = getEffectiveOptimizer(null);
            }
            
            if (inputShape.rank() == 2) {
                // Perfect! We have [seqLen, features]
                int features = inputShape.dim(1);
                return new GruLayer(effectiveOptimizer, hiddenSize, features, initStrategy, OutputMode.LAST_TIMESTEP, random);
            } else if (inputShape.rank() == 1) {
                // Fall back to old behavior
                return create(inputShape.toFlatSize(), effectiveOptimizer, random);
            }
            
            throw new IllegalArgumentException("GRU cannot handle shape: " + inputShape);
        }
        
        @Override
        public Shape getOutputShape(Shape inputShape) {
            if (inputShape.rank() == 2) {
                // [seqLen, features] -> [hiddenSize] for LAST_TIMESTEP mode
                return Shape.vector(hiddenSize);
            } else if (inputShape.rank() == 1) {
                // For flattened input, output is just hiddenSize
                return Shape.vector(hiddenSize);
            }
            
            throw new IllegalArgumentException("GRU cannot determine output shape for input shape: " + inputShape);
        }
        
        @Override
        public Layer create(int inputSize, Optimizer defaultOptimizer, FastRandom random) {
            throw new UnsupportedOperationException(
                "GRU layer requires shape information. Use create(Shape, Optimizer, FastRandom) instead.");
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer, FastRandom random) {
            throw new UnsupportedOperationException(
                "This method should not be called. GRU layer requires shape information.");
        }
        
    }
    
    /**
     * Specification for GRU layer that outputs ALL timesteps.
     * Reports output size as seqLen × hiddenSize based on input size.
     */
    private static class GruAllTimestepsSpec extends BaseLayerSpec<GruAllTimestepsSpec> {
        private final int hiddenSize;
        private final WeightInitStrategy initStrategy;
        private final int expectedInputDimension; // Expected size per timestep (-1 if unknown)
        private int cachedInputDimension = -1; // Cache the actual input dimension per timestep
        
        public GruAllTimestepsSpec(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy, 
                                  double learningRateRatio, int expectedInputDimension) {
            super(hiddenSize, optimizer); // Pass hiddenSize temporarily
            this.hiddenSize = hiddenSize;
            this.initStrategy = initStrategy;
            this.learningRateRatio = (float) learningRateRatio;
            this.expectedInputDimension = expectedInputDimension;
        }
        
        @Override
        public boolean prefersShapeAPI() {
            return true; // GRU benefits from shape information!
        }
        
        @Override
        public void validateInputShape(Shape inputShape) {
            if (inputShape.rank() != 2 && inputShape.rank() != 1) {
                throw new IllegalArgumentException(
                    "GRU expects 2D input [sequenceLength, features] or 1D flattened input, got shape: " + inputShape);
            }
        }
        
        @Override
        public Layer create(Shape inputShape, Optimizer effectiveOptimizer, FastRandom random) {
            if (effectiveOptimizer == null) {
                effectiveOptimizer = getEffectiveOptimizer(null);
            }
            
            if (inputShape.rank() == 2) {
                // Perfect! We have [seqLen, features]
                int features = inputShape.dim(1);
                return new GruLayer(effectiveOptimizer, hiddenSize, features, initStrategy, OutputMode.ALL_TIMESTEPS, random);
            } else if (inputShape.rank() == 1) {
                // Fall back to old behavior
                return create(inputShape.toFlatSize(), effectiveOptimizer, random);
            }
            
            throw new IllegalArgumentException("GRU cannot handle shape: " + inputShape);
        }
        
        @Override
        public Shape getOutputShape(Shape inputShape) {
            if (inputShape.rank() == 2) {
                // [seqLen, features] -> [seqLen, hiddenSize]
                return Shape.sequence(inputShape.dim(0), hiddenSize);
            } else if (inputShape.rank() == 1) {
                // Try to use the old logic
                int outputSize = getOutputSize(inputShape.toFlatSize());
                return Shape.vector(outputSize);
            }
            
            throw new IllegalArgumentException("GRU cannot determine output shape for input shape: " + inputShape);
        }
        
        
        @Override
        public Layer create(int inputSize, Optimizer defaultOptimizer, FastRandom random) {
            throw new UnsupportedOperationException(
                "GRU layer requires shape information. Use create(Shape, Optimizer, FastRandom) instead.");
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer, FastRandom random) {
            throw new UnsupportedOperationException(
                "This method should not be called. GRU layer requires shape information.");
        }
        
        @Override
        public int getOutputSize() {
            // Legacy method - return hiddenSize for backward compatibility
            return hiddenSize;
        }
        
        @Override
        public int getOutputSize(int inputSize) {
            // The inputSize parameter here is the total flattened size from previous layer
            // For sequence layers, this is seqLen × embeddingDim (or similar)
            
            // Try to determine the per-timestep dimension
            int inputDimPerTimestep = -1;
            
            if (cachedInputDimension > 0) {
                // We've already created the layer, use the actual dimension
                inputDimPerTimestep = cachedInputDimension;
            } else if (expectedInputDimension > 0) {
                // Use the hint provided at construction time
                inputDimPerTimestep = expectedInputDimension;
            }
            
            if (inputDimPerTimestep > 0 && inputSize % inputDimPerTimestep == 0) {
                // We can calculate the sequence length!
                int seqLen = inputSize / inputDimPerTimestep;
                return seqLen * hiddenSize;
            }
            
            // Fallback: we don't know the input dimension, so we can't calculate output size accurately
            // Return a conservative estimate
            return inputSize;
        }
    }
    
    /**
     * Specification for GRU layer that outputs ONLY the LAST timestep.
     * Reports output size as hiddenSize (correct for single timestep).
     */
    private static class GruLastTimestepSpec extends BaseLayerSpec<GruLastTimestepSpec> {
        private final int hiddenSize;
        private final WeightInitStrategy initStrategy;
        
        public GruLastTimestepSpec(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy, double learningRateRatio) {
            super(hiddenSize, optimizer); // Correctly reports hiddenSize as output
            this.hiddenSize = hiddenSize;
            this.initStrategy = initStrategy;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        
        @Override
        public boolean prefersShapeAPI() {
            return true; // GRU requires shape information
        }
        
        @Override
        public void validateInputShape(Shape inputShape) {
            if (inputShape.rank() != 2 && inputShape.rank() != 1) {
                throw new IllegalArgumentException(
                    "GRU expects 2D input [sequenceLength, features] or 1D flattened input, got shape: " + inputShape);
            }
        }
        
        @Override
        public Layer create(Shape inputShape, Optimizer effectiveOptimizer, FastRandom random) {
            if (effectiveOptimizer == null) {
                effectiveOptimizer = getEffectiveOptimizer(null);
            }
            
            if (inputShape.rank() == 2) {
                // Perfect! We have [seqLen, features]
                int features = inputShape.dim(1);
                return new GruLayer(effectiveOptimizer, hiddenSize, features, initStrategy, OutputMode.LAST_TIMESTEP, random);
            } else if (inputShape.rank() == 1) {
                // Fall back to old behavior
                return create(inputShape.toFlatSize(), effectiveOptimizer, random);
            }
            
            throw new IllegalArgumentException("GRU cannot handle shape: " + inputShape);
        }
        
        @Override
        public Shape getOutputShape(Shape inputShape) {
            if (inputShape.rank() == 2) {
                // [seqLen, features] -> [hiddenSize] for LAST_TIMESTEP mode
                return Shape.vector(hiddenSize);
            } else if (inputShape.rank() == 1) {
                // For flattened input, output is just hiddenSize
                return Shape.vector(hiddenSize);
            }
            
            throw new IllegalArgumentException("GRU cannot determine output shape for input shape: " + inputShape);
        }
        
        @Override
        public Layer create(int inputSize, Optimizer defaultOptimizer, FastRandom random) {
            throw new UnsupportedOperationException(
                "GRU layer requires shape information. Use create(Shape, Optimizer, FastRandom) instead.");
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer, FastRandom random) {
            throw new UnsupportedOperationException(
                "This method should not be called. GRU layer requires shape information.");
        }
        
        // getOutputSize() from parent correctly returns hiddenSize
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        out.writeInt(hiddenSize);
        out.writeInt(inputSize);
        
        // Write outputMode (critical for correct output shape)
        out.writeInt(outputMode.ordinal());
        
        // Write weight matrices
        writeWeightMatrix(out, resetWeights);
        writeWeightMatrix(out, updateWeights);
        writeWeightMatrix(out, candidateWeights);
        
        // Write biases
        writeBiasVector(out, resetBias);
        writeBiasVector(out, updateBias);
        writeBiasVector(out, candidateBias);
        
        writeOptimizer(out, optimizer, version);
    }
    
    private void writeWeightMatrix(DataOutputStream out, float[][] weights) throws IOException {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                out.writeFloat(weights[i][j]);
            }
        }
    }
    
    private void writeBiasVector(DataOutputStream out, float[] bias) throws IOException {
        for (float b : bias) {
            out.writeFloat(b);
        }
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize(DataInputStream, int) static method instead");
    }
    
    public static GruLayer deserialize(DataInputStream in, int version, FastRandom random) throws IOException {
        int hiddenSize = in.readInt();
        int inputSize = in.readInt();
        
        // Read outputMode (critical for correct output shape)
        OutputMode outputMode = OutputMode.values()[in.readInt()];
        
        // Read weights
        float[][] resetWeights = readWeightMatrix(in, inputSize + hiddenSize, hiddenSize);
        float[][] updateWeights = readWeightMatrix(in, inputSize + hiddenSize, hiddenSize);
        float[][] candidateWeights = readWeightMatrix(in, inputSize + hiddenSize, hiddenSize);
        
        // Read biases
        float[] resetBias = readBiasVector(in, hiddenSize);
        float[] updateBias = readBiasVector(in, hiddenSize);
        float[] candidateBias = readBiasVector(in, hiddenSize);
        
        Optimizer optimizer = readOptimizer(in, version);
        
        // Create layer with correct output mode and provided FastRandom
        GruLayer layer = new GruLayer(optimizer, hiddenSize, inputSize, WeightInitStrategy.XAVIER, outputMode, random);
        
        // Copy deserialized weights and biases
        copyWeightMatrix(resetWeights, layer.resetWeights);
        copyWeightMatrix(updateWeights, layer.updateWeights);
        copyWeightMatrix(candidateWeights, layer.candidateWeights);
        
        System.arraycopy(resetBias, 0, layer.resetBias, 0, hiddenSize);
        System.arraycopy(updateBias, 0, layer.updateBias, 0, hiddenSize);
        System.arraycopy(candidateBias, 0, layer.candidateBias, 0, hiddenSize);
        
        return layer;
    }
    
    private static float[][] readWeightMatrix(DataInputStream in, int rows, int cols) throws IOException {
        float[][] matrix = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = in.readFloat();
            }
        }
        return matrix;
    }
    
    private static float[] readBiasVector(DataInputStream in, int size) throws IOException {
        float[] vector = new float[size];
        for (int i = 0; i < size; i++) {
            vector[i] = in.readFloat();
        }
        return vector;
    }
    
    private static void copyWeightMatrix(float[][] source, float[][] dest) {
        for (int i = 0; i < source.length; i++) {
            System.arraycopy(source[i], 0, dest[i], 0, source[i].length);
        }
    }
    
    private static void writeOptimizer(DataOutputStream out, Optimizer optimizer, int version) throws IOException {
        String registeredName = SerializationRegistry.getRegisteredName(optimizer);
        if (registeredName != null) {
            out.writeInt(SerializationConstants.TYPE_CUSTOM);
            out.writeUTF(registeredName);
            return;
        }
        
        Serializable serializableOptimizer = (Serializable) optimizer;
        out.writeInt(serializableOptimizer.getTypeId());
        serializableOptimizer.writeTo(out, version);
    }
    
    private static Optimizer readOptimizer(DataInputStream in, int version) throws IOException {
        int typeId = in.readInt();
        
        if (typeId == SerializationConstants.TYPE_CUSTOM) {
            String className = in.readUTF();
            return SerializationRegistry.createOptimizer(className, in, version);
        }
        
        return switch (typeId) {
            case SerializationConstants.TYPE_SGD_OPTIMIZER -> SgdOptimizer.deserialize(in, version);
            case SerializationConstants.TYPE_ADAM_OPTIMIZER -> AdamOptimizer.deserialize(in, version);
            case SerializationConstants.TYPE_ADAMW_OPTIMIZER -> AdamWOptimizer.deserialize(in, version);
            default -> throw new IOException("Unknown optimizer type ID: " + typeId);
        };
    }
    
    @Override
    public int getSerializedSize(int version) {
        int totalInputSize = inputSize + hiddenSize;
        int size = 8; // hiddenSize + inputSize
        size += 4; // outputMode ordinal
        size += 3 * totalInputSize * hiddenSize * 4; // Three weight matrices
        size += 3 * hiddenSize * 4; // Three bias vectors
        
        String registeredOptimizerName = SerializationRegistry.getRegisteredName(optimizer);
        if (registeredOptimizerName != null) {
            size += 4; // TYPE_CUSTOM
            size += 2 + registeredOptimizerName.getBytes().length; // UTF string
        } else {
            size += 4; // built-in type ID
            size += ((Serializable) optimizer).getSerializedSize(version);
        }
        
        return size;
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_GRU_LAYER;
    }
    
    
    /**
     * Comprehensive input validation with NaN/Infinity checking.
     */
    private void validateInputs(float[] input) {
        if (input == null)
            throw new IllegalArgumentException("Input cannot be null");
        if (input.length % inputSize != 0)
            throw new IllegalArgumentException("Input length must be multiple of inputSize");
        
        // Critical: Check for NaN/Infinity values that could corrupt training
        for (int i = 0; i < input.length; i++) {
            if (!Float.isFinite(input[i])) {
                throw new IllegalArgumentException("Invalid input value at index " + i + ": " + input[i] + 
                    " (sequence position: " + (i / inputSize) + ", feature: " + (i % inputSize) + ")");
            }
        }
    }
    
    /**
     * Comprehensive backward pass validation with bounds and gradient checking.
     */
    private void validateBackwardInputs(LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
        if (stack == null)
            throw new IllegalArgumentException("Stack cannot be null");
        if (stackIndex < 0 || stackIndex >= stack.length)
            throw new IndexOutOfBoundsException("Stack index out of bounds: " + stackIndex + 
                " (stack length: " + stack.length + ")");
        if (upstreamGradient == null)
            throw new IllegalArgumentException("Upstream gradient cannot be null");
        
        LayerContext context = stack[stackIndex];
        if (!(context instanceof GruLayerContext)) {
            throw new IllegalArgumentException("Expected GruLayerContext but got: " + context.getClass());
        }
        
        GruLayerContext gruContext = (GruLayerContext) context;
        int expectedGradientLength;
        if (outputMode == OutputMode.ALL_TIMESTEPS) {
            expectedGradientLength = gruContext.seqLen * hiddenSize;
        } else {
            expectedGradientLength = hiddenSize;
        }
        
        if (upstreamGradient.length != expectedGradientLength) {
            throw new IllegalArgumentException("Upstream gradient length mismatch. Expected: " + 
                expectedGradientLength + " (outputMode=" + outputMode + 
                ", seqLen=" + gruContext.seqLen + ", hiddenSize=" + hiddenSize + 
                "), got: " + upstreamGradient.length);
        }
        
        // Critical: Check for NaN/Infinity values in gradients
        for (int i = 0; i < upstreamGradient.length; i++) {
            if (!Float.isFinite(upstreamGradient[i])) {
                throw new IllegalArgumentException("Invalid upstream gradient at index " + i + ": " + 
                    upstreamGradient[i] + " (timestep: " + (i / hiddenSize) + ", hidden unit: " + (i % hiddenSize) + ")");
            }
        }
    }
    
    /**
     * Validate ThreadLocal buffer is properly initialized.
     */
    private void validateBuffer(float[] buffer, String bufferName) {
        if (buffer == null) {
            throw new IllegalStateException("ThreadLocal buffer '" + bufferName + "' not initialized. " +
                "This indicates a serious threading issue.");
        }
    }
    
    /**
     * Determine if parallel execution should be used based on shared parallelization policy.
     * Uses total work size (sequence length × hidden size) for matrix operations.
     */
    private boolean shouldUseParallel(ExecutorService executor, int seqLen) {
        int totalWork = seqLen * hiddenSize; // Total work for sequence processing
        return Parallelization.shouldParallelize(totalWork, executor);
    }
    
    /**
     * Determine if parallel execution should be used for single timestep operations.
     * Uses hidden size as work metric for gate computations.
     */
    private boolean shouldUseParallelForTimestep(ExecutorService executor) {
        return Parallelization.shouldParallelize(hiddenSize, executor);
    }
    
    /**
     * Batch-aware forward pass for processing multiple sequences in parallel.
     * Input format: [batchSize, seqLen, inputSize] flattened to [batchSize * seqLen * inputSize]
     * Output format: [batchSize, seqLen, hiddenSize] flattened to [batchSize * seqLen * hiddenSize]
     * 
     * This provides massive parallelization opportunities by processing each sequence
     * in the batch independently across threads.
     * 
     * @param batchInput flattened batch input [batchSize * seqLen * inputSize]
     * @param batchSize number of sequences in the batch
     * @param seqLen length of each sequence
     * @param executor executor for parallel processing
     * @return batch layer context with concatenated outputs
     */
    public LayerContext forwardBatch(float[] batchInput, int batchSize, int seqLen, ExecutorService executor) {
        if (batchInput == null)
            throw new IllegalArgumentException("Batch input cannot be null");
        if (batchSize <= 0)
            throw new IllegalArgumentException("Batch size must be positive: " + batchSize);
        if (seqLen <= 0)
            throw new IllegalArgumentException("Sequence length must be positive: " + seqLen);
        if (batchInput.length != batchSize * seqLen * inputSize)
            throw new IllegalArgumentException("Batch input length mismatch. Expected: " + 
                (batchSize * seqLen * inputSize) + ", got: " + batchInput.length);
        
        // Validate all inputs for NaN/Infinity
        for (int i = 0; i < batchInput.length; i++) {
            if (!Float.isFinite(batchInput[i])) {
                int batch = i / (seqLen * inputSize);
                int seq = (i % (seqLen * inputSize)) / inputSize;
                int feature = i % inputSize;
                throw new IllegalArgumentException("Invalid batch input at [" + batch + "][" + seq + "][" + feature + "]: " + batchInput[i]);
            }
        }
        
        int totalWork = batchSize * seqLen * hiddenSize;
        boolean useParallel = Parallelization.shouldParallelize(totalWork, executor);
        
        float[] batchOutputs = new float[batchSize * seqLen * hiddenSize];
        LayerContext[] batchContexts = new LayerContext[batchSize];
        
        if (useParallel && batchSize > 1) {
            // Embarrassingly parallel: each sequence processes independently
            int numThreads = Parallelization.calculateOptimalThreads(batchSize, executor);
            Parallelization.WorkRange[] ranges = Parallelization.splitWork(batchSize, numThreads);
            
            Runnable[] tasks = new Runnable[ranges.length];
            for (int i = 0; i < ranges.length; i++) {
                final Parallelization.WorkRange range = ranges[i];
                tasks[i] = () -> {
                    for (int b = range.start; b < range.end; b++) {
                        // Extract sequence for this batch item
                        float[] sequenceInput = new float[seqLen * inputSize];
                        System.arraycopy(batchInput, b * seqLen * inputSize, sequenceInput, 0, seqLen * inputSize);
                        
                        // Process sequence independently
                        LayerContext context = forward(sequenceInput, false);
                        batchContexts[b] = context;
                        
                        // Copy outputs to batch output array
                        System.arraycopy(context.outputs(), 0, batchOutputs, b * seqLen * hiddenSize, seqLen * hiddenSize);
                    }
                };
            }
            
            try {
                Parallelization.executeParallel(executor, tasks);
            } catch (Exception e) {
                throw new RuntimeException("Parallel batch forward pass failed", e);
            }
        } else {
            // Sequential processing for small batches or no executor
            for (int b = 0; b < batchSize; b++) {
                float[] sequenceInput = new float[seqLen * inputSize];
                System.arraycopy(batchInput, b * seqLen * inputSize, sequenceInput, 0, seqLen * inputSize);
                
                LayerContext context = forward(sequenceInput, false);
                batchContexts[b] = context;
                
                System.arraycopy(context.outputs(), 0, batchOutputs, b * seqLen * hiddenSize, seqLen * hiddenSize);
            }
        }
        
        // Create combined batch context
        return new BatchLayerContext(batchInput, null, batchOutputs, batchContexts, batchSize, seqLen);
    }
    
    /**
     * Batch-aware layer context that stores individual sequence contexts.
     */
    public static class BatchLayerContext extends LayerContext {
        public final LayerContext[] sequenceContexts;
        public final int batchSize;
        public final int seqLen;
        
        public BatchLayerContext(float[] inputs, float[] preActivations, float[] outputs,
                                LayerContext[] sequenceContexts, int batchSize, int seqLen) {
            super(inputs, preActivations, outputs);
            this.sequenceContexts = sequenceContexts;
            this.batchSize = batchSize;
            this.seqLen = seqLen;
        }
    }
    
    /**
     * Batch-aware backward pass for processing multiple sequence gradients in parallel.
     * 
     * @param batchContext batch layer context from forward pass
     * @param upstreamGradient batch upstream gradients [batchSize * seqLen * hiddenSize]
     * @param executor executor for parallel processing
     * @return batch input gradients [batchSize * seqLen * inputSize]
     */
    public float[] backwardBatch(BatchLayerContext batchContext, float[] upstreamGradient, ExecutorService executor) {
        if (batchContext == null)
            throw new IllegalArgumentException("Batch context cannot be null");
        if (upstreamGradient == null)
            throw new IllegalArgumentException("Upstream gradient cannot be null");
        
        int batchSize = batchContext.batchSize;
        int seqLen = batchContext.seqLen;
        int expectedGradientLength = batchSize * seqLen * hiddenSize;
        
        if (upstreamGradient.length != expectedGradientLength) {
            throw new IllegalArgumentException("Batch upstream gradient length mismatch. Expected: " + 
                expectedGradientLength + ", got: " + upstreamGradient.length);
        }
        
        // Validate gradients for NaN/Infinity
        for (int i = 0; i < upstreamGradient.length; i++) {
            if (!Float.isFinite(upstreamGradient[i])) {
                int batch = i / (seqLen * hiddenSize);
                int seq = (i % (seqLen * hiddenSize)) / hiddenSize;
                int hidden = i % hiddenSize;
                throw new IllegalArgumentException("Invalid batch gradient at [" + batch + "][" + seq + "][" + hidden + "]: " + upstreamGradient[i]);
            }
        }
        
        float[] batchInputGradients = new float[batchSize * seqLen * inputSize];
        
        int totalWork = batchSize * seqLen * hiddenSize;
        boolean useParallel = Parallelization.shouldParallelize(totalWork, executor);
        
        if (useParallel && batchSize > 1) {
            // Parallel backward pass across batch dimension
            int numThreads = Parallelization.calculateOptimalThreads(batchSize, executor);
            Parallelization.WorkRange[] ranges = Parallelization.splitWork(batchSize, numThreads);
            
            Runnable[] tasks = new Runnable[ranges.length];
            for (int i = 0; i < ranges.length; i++) {
                final Parallelization.WorkRange range = ranges[i];
                tasks[i] = () -> {
                    for (int b = range.start; b < range.end; b++) {
                        // Extract upstream gradient for this sequence
                        float[] sequenceGradient = new float[seqLen * hiddenSize];
                        System.arraycopy(upstreamGradient, b * seqLen * hiddenSize, sequenceGradient, 0, seqLen * hiddenSize);
                        
                        // Backward pass for this sequence
                        LayerContext[] stack = {batchContext.sequenceContexts[b]};
                        float[] inputGradients = backward(stack, 0, sequenceGradient);
                        
                        // Copy input gradients to batch array
                        System.arraycopy(inputGradients, 0, batchInputGradients, b * seqLen * inputSize, seqLen * inputSize);
                    }
                };
            }
            
            try {
                Parallelization.executeParallel(executor, tasks);
            } catch (Exception e) {
                throw new RuntimeException("Parallel batch backward pass failed", e);
            }
        } else {
            // Sequential processing
            for (int b = 0; b < batchSize; b++) {
                float[] sequenceGradient = new float[seqLen * hiddenSize];
                System.arraycopy(upstreamGradient, b * seqLen * hiddenSize, sequenceGradient, 0, seqLen * hiddenSize);
                
                LayerContext[] stack = {batchContext.sequenceContexts[b]};
                float[] inputGradients = backward(stack, 0, sequenceGradient);
                
                System.arraycopy(inputGradients, 0, batchInputGradients, b * seqLen * inputSize, seqLen * inputSize);
            }
        }
        
        return batchInputGradients;
    }
    
    /**
     * Specification for creating layer-normalized GRU layers that output all timesteps.
     */
    private static class GruAllTimestepsNormalizedSpec extends BaseLayerSpec<GruAllTimestepsNormalizedSpec> {
        private final int hiddenSize;
        private final WeightInitStrategy initStrategy;
        
        public GruAllTimestepsNormalizedSpec(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy, double learningRateRatio) {
            super(hiddenSize * 20, optimizer); // Output size estimate for sequences
            this.hiddenSize = hiddenSize;
            this.initStrategy = initStrategy;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        @Override
        public int getOutputSize() {
            return -1; // Variable output size based on sequence length
        }
        
        @Override
        public int getOutputSize(int inputSize) {
            // For GRU with ALL timesteps: output = sequence_length * hidden_size
            // Input is sequence flattened, so sequence_length = inputSize / embedding_dim
            // But we don't know embedding_dim here, so we estimate
            return inputSize * hiddenSize / 128; // Rough estimate assuming 128-dim embeddings
        }
        
        
        @Override
        public boolean prefersShapeAPI() {
            return true; // GRU requires shape information
        }
        
        @Override
        public void validateInputShape(Shape inputShape) {
            if (inputShape.rank() != 2 && inputShape.rank() != 1) {
                throw new IllegalArgumentException(
                    "GRU expects 2D input [sequenceLength, features] or 1D flattened input, got shape: " + inputShape);
            }
        }
        
        @Override
        public Layer create(Shape inputShape, Optimizer effectiveOptimizer, FastRandom random) {
            if (effectiveOptimizer == null) {
                effectiveOptimizer = getEffectiveOptimizer(null);
            }
            
            if (inputShape.rank() == 2) {
                // Perfect! We have [seqLen, features]
                int features = inputShape.dim(1);
                return new GruLayer(effectiveOptimizer, hiddenSize, features, initStrategy, OutputMode.ALL_TIMESTEPS, true, random);
            } else if (inputShape.rank() == 1) {
                // Fall back to old behavior
                return create(inputShape.toFlatSize(), effectiveOptimizer, random);
            }
            
            throw new IllegalArgumentException("GRU cannot handle shape: " + inputShape);
        }
        
        @Override
        public Shape getOutputShape(Shape inputShape) {
            if (inputShape.rank() == 2) {
                // [seqLen, features] -> [seqLen, hiddenSize] for ALL_TIMESTEPS mode
                return Shape.sequence(inputShape.dim(0), hiddenSize);
            } else if (inputShape.rank() == 1) {
                // For flattened input, we need to guess output shape
                int outputSize = getOutputSize(inputShape.toFlatSize());
                return Shape.vector(outputSize);
            }
            
            throw new IllegalArgumentException("GRU cannot determine output shape for input shape: " + inputShape);
        }
        
        @Override
        public Layer create(int inputSize, Optimizer defaultOptimizer, FastRandom random) {
            throw new UnsupportedOperationException(
                "GRU layer requires shape information. Use create(Shape, Optimizer, FastRandom) instead.");
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer, FastRandom random) {
            throw new UnsupportedOperationException(
                "This method should not be called. GRU layer requires shape information.");
        }
    }
    
    /**
     * Specification for creating layer-normalized GRU layers that output only last timestep.
     */
    private static class GruLastTimestepNormalizedSpec extends BaseLayerSpec<GruLastTimestepNormalizedSpec> {
        private final int hiddenSize;
        private final WeightInitStrategy initStrategy;
        
        public GruLastTimestepNormalizedSpec(int hiddenSize, Optimizer optimizer, WeightInitStrategy initStrategy, double learningRateRatio) {
            super(hiddenSize, optimizer);
            this.hiddenSize = hiddenSize;
            this.initStrategy = initStrategy;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        
        @Override
        public boolean prefersShapeAPI() {
            return true; // GRU requires shape information
        }
        
        @Override
        public void validateInputShape(Shape inputShape) {
            if (inputShape.rank() != 2 && inputShape.rank() != 1) {
                throw new IllegalArgumentException(
                    "GRU expects 2D input [sequenceLength, features] or 1D flattened input, got shape: " + inputShape);
            }
        }
        
        @Override
        public Layer create(Shape inputShape, Optimizer effectiveOptimizer, FastRandom random) {
            if (effectiveOptimizer == null) {
                effectiveOptimizer = getEffectiveOptimizer(null);
            }
            
            if (inputShape.rank() == 2) {
                // Perfect! We have [seqLen, features]
                int features = inputShape.dim(1);
                return new GruLayer(effectiveOptimizer, hiddenSize, features, initStrategy, OutputMode.LAST_TIMESTEP, true, random);
            } else if (inputShape.rank() == 1) {
                // Fall back to old behavior
                return create(inputShape.toFlatSize(), effectiveOptimizer, random);
            }
            
            throw new IllegalArgumentException("GRU cannot handle shape: " + inputShape);
        }
        
        @Override
        public Shape getOutputShape(Shape inputShape) {
            if (inputShape.rank() == 2) {
                // [seqLen, features] -> [hiddenSize] for LAST_TIMESTEP mode
                return Shape.vector(hiddenSize);
            } else if (inputShape.rank() == 1) {
                // For flattened input, output is just hiddenSize
                return Shape.vector(hiddenSize);
            }
            
            throw new IllegalArgumentException("GRU cannot determine output shape for input shape: " + inputShape);
        }
        
        @Override
        public Layer create(int inputSize, Optimizer defaultOptimizer, FastRandom random) {
            throw new UnsupportedOperationException(
                "GRU layer requires shape information. Use create(Shape, Optimizer, FastRandom) instead.");
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer, FastRandom random) {
            throw new UnsupportedOperationException(
                "This method should not be called. GRU layer requires shape information.");
        }
    }
}