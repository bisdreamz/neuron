package dev.neuronic.net.layers;

import dev.neuronic.net.Shape;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.activators.SigmoidActivator;
import dev.neuronic.net.activators.TanhActivator;
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
        
        public GruLayerContext(float[] inputs, float[] preActivations, float[] outputs, int seqLen,
                             float[][] resetGates, float[][] updateGates, float[][] candidates,
                             float[][] hiddenStates, float[][] concatenatedInputs) {
            super(inputs, preActivations, outputs);
            this.seqLen = seqLen;
            this.resetGates = resetGates;
            this.updateGates = updateGates;
            this.candidates = candidates;
            this.hiddenStates = hiddenStates;
            this.concatenatedInputs = concatenatedInputs;
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
        final float[] outputBuffer;             // expandable
        
        // Gradient buffers for backward pass
        final float[][] resetWeightGradients;
        final float[][] updateWeightGradients;
        final float[][] candidateWeightGradients;
        
        // Additional temporary buffers for backward pass optimization
        final float[] tempBuffer1;              // hiddenSize
        final float[] tempBuffer2;              // hiddenSize  
        final float[] tempConcatBuffer;         // totalInputSize
        
        // Sequence-level buffers that grow as needed
        float[][] resetGates;                   // [seqLen][hiddenSize]
        float[][] updateGates;                  // [seqLen][hiddenSize]
        float[][] candidates;                   // [seqLen][hiddenSize]
        float[][] hiddenStates;                 // [seqLen+1][hiddenSize]
        float[][] concatenatedInputs;           // [seqLen][totalInputSize]
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
            this.outputBuffer = new float[hiddenSize * 64]; // Initial capacity
            
            // Gradient buffers
            this.resetWeightGradients = new float[totalInputSize][hiddenSize];
            this.updateWeightGradients = new float[totalInputSize][hiddenSize];
            this.candidateWeightGradients = new float[totalInputSize][hiddenSize];
            
            // Additional temporary buffers
            this.tempBuffer1 = new float[hiddenSize];
            this.tempBuffer2 = new float[hiddenSize];
            this.tempConcatBuffer = new float[totalInputSize];
        }
        
        /**
         * Ensure output buffer has sufficient capacity, expanding if necessary.
         */
        float[] ensureOutputCapacity(int requiredSize) {
            if (outputBuffer.length < requiredSize) {
                // Expand capacity to next power of 2 or required size, whichever is larger
                int newCapacity = Math.max(requiredSize, Integer.highestOneBit(outputBuffer.length) << 1);
                return new float[newCapacity];
            }
            return outputBuffer;
        }
        
        /**
         * Ensure sequence-level buffers have sufficient capacity for the given sequence length.
         */
        void ensureSequenceCapacity(int seqLen, int hiddenSize, int totalInputSize) {
            if (currentSequenceCapacity < seqLen) {
                // Allocate with some extra capacity to avoid frequent reallocations
                int newCapacity = Math.max(seqLen, currentSequenceCapacity * 2);
                
                resetGates = new float[newCapacity][hiddenSize];
                updateGates = new float[newCapacity][hiddenSize];
                candidates = new float[newCapacity][hiddenSize];
                hiddenStates = new float[newCapacity + 1][hiddenSize]; // +1 for initial h_0
                concatenatedInputs = new float[newCapacity][totalInputSize];
                
                currentSequenceCapacity = newCapacity;
            }
        }
    }
    
    // Single ThreadLocal for all buffers - major performance improvement
    private final ThreadLocal<GruBuffers> allBuffers;
    
    public GruLayer(Optimizer optimizer, int hiddenSize, int inputSize, WeightInitStrategy initStrategy) {
        this(optimizer, hiddenSize, inputSize, initStrategy, OutputMode.ALL_TIMESTEPS);
    }
    
    public GruLayer(Optimizer optimizer, int hiddenSize, int inputSize, WeightInitStrategy initStrategy, OutputMode outputMode) {
        this(optimizer, hiddenSize, inputSize, initStrategy, outputMode, false);
    }
    
    public GruLayer(Optimizer optimizer, int hiddenSize, int inputSize, WeightInitStrategy initStrategy, OutputMode outputMode, boolean useLayerNorm) {
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
        
        initializeWeights(initStrategy);
    }
    
    private void initializeWeights(WeightInitStrategy strategy) {
        switch (strategy) {
            case XAVIER -> {
                NetMath.weightInitXavier(resetWeights, inputSize + hiddenSize, hiddenSize);
                NetMath.weightInitXavier(updateWeights, inputSize + hiddenSize, hiddenSize);
                NetMath.weightInitXavier(candidateWeights, inputSize + hiddenSize, hiddenSize);
            }
            case HE -> {
                NetMath.weightInitHe(resetWeights, inputSize + hiddenSize);
                NetMath.weightInitHe(updateWeights, inputSize + hiddenSize);
                NetMath.weightInitHe(candidateWeights, inputSize + hiddenSize);
            }
        }
        
        NetMath.biasInit(resetBias, 0.0f);
        NetMath.biasInit(updateBias, 0.0f);
        NetMath.biasInit(candidateBias, 0.0f);
    }
    
    @Override
    public LayerContext forward(float[] input) {
        validateInputs(input);
        
        int seqLen = input.length / inputSize;
        if (seqLen == 0)
            throw new IllegalArgumentException("Sequence length cannot be zero");
        
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
        java.util.Arrays.fill(hiddenStates[0], 0.0f);
        
        // Process sequence timestep by timestep
        for (int t = 0; t < seqLen; t++) {
            // Extract current timestep input using efficient array copying
            System.arraycopy(input, t * inputSize, buffers.currentInputBuffer, 0, inputSize);
            
            // Forward through GRU cell and store all intermediate states
            forwardCellWithStorageOptimized(buffers.currentInputBuffer, hiddenStates[t], hiddenStates[t + 1], 
                                          resetGates[t], updateGates[t], candidates[t], concatenatedInputs[t], buffers);
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
        
        // Copy data from ThreadLocal buffers to fresh arrays
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(resetGates[t], 0, freshResetGates[t], 0, hiddenSize);
            System.arraycopy(updateGates[t], 0, freshUpdateGates[t], 0, hiddenSize);
            System.arraycopy(candidates[t], 0, freshCandidates[t], 0, hiddenSize);
            System.arraycopy(hiddenStates[t + 1], 0, freshHiddenStates[t + 1], 0, hiddenSize);
            System.arraycopy(concatenatedInputs[t], 0, freshConcatenatedInputs[t], 0, totalInputSize);
        }
        // Copy initial hidden state h_0
        System.arraycopy(hiddenStates[0], 0, freshHiddenStates[0], 0, hiddenSize);
        
        // Use safe factory method for ThreadLocal buffers - creates a custom context with input copy
        return new GruLayerContext(input.clone(), null, outputs, seqLen, 
                                 freshResetGates, freshUpdateGates, freshCandidates, freshHiddenStates, freshConcatenatedInputs);
    }
    
    
    /**
     * Optimized forward pass through GRU cell using pre-allocated buffers.
     * Eliminates all dynamic allocation for maximum performance.
     * Optionally applies layer normalization to gate pre-activations.
     */
    private void forwardCellWithStorageOptimized(float[] input, float[] prevHidden, float[] newHidden,
                                               float[] resetGate, float[] updateGate, float[] candidate,
                                               float[] concatenatedInput, GruBuffers buffers) {
        // Concatenate input and previous hidden state
        System.arraycopy(input, 0, concatenatedInput, 0, inputSize);
        System.arraycopy(prevHidden, 0, concatenatedInput, inputSize, hiddenSize);
        
        // Reset gate: r_t = σ(W_r * [h_{t-1}, x_t] + b_r)
        NetMath.matrixPreActivationsColumnMajor(concatenatedInput, resetWeights, resetBias, resetGate);
        if (useLayerNorm && resetLayerNorm != null) {
            resetLayerNorm.forward(resetGate, resetGate);
        }
        SigmoidActivator.INSTANCE.activate(resetGate, resetGate); // Vectorized internally
        
        // Update gate: z_t = σ(W_z * [h_{t-1}, x_t] + b_z)  
        NetMath.matrixPreActivationsColumnMajor(concatenatedInput, updateWeights, updateBias, updateGate);
        if (useLayerNorm && updateLayerNorm != null) {
            updateLayerNorm.forward(updateGate, updateGate);
        }
        SigmoidActivator.INSTANCE.activate(updateGate, updateGate); // Vectorized internally
        
        // Prepare input for candidate: [r_t ⊙ h_{t-1}, x_t] using consolidated buffers
        float[] resetHiddenProduct = buffers.tempBuffer1; // Reuse temp buffer (hiddenSize)
        NetMath.elementwiseMultiply(resetGate, prevHidden, resetHiddenProduct);
        System.arraycopy(resetHiddenProduct, 0, concatenatedInput, inputSize, hiddenSize);
        
        // Candidate: h̃_t = tanh(W_h * [r_t ⊙ h_{t-1}, x_t] + b_h)
        NetMath.matrixPreActivationsColumnMajor(concatenatedInput, candidateWeights, candidateBias, candidate);
        if (useLayerNorm && candidateLayerNorm != null) {
            candidateLayerNorm.forward(candidate, candidate);
        }
        TanhActivator.INSTANCE.activate(candidate, candidate);
        
        // Final hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t using consolidated buffers
        float[] oneMinusUpdate = buffers.tempBuffer1;      // Reuse temp buffer (hiddenSize)
        float[] term1 = buffers.tempBuffer2;               // Reuse temp buffer (hiddenSize) 
        float[] term2 = buffers.hiddenStateBuffer;         // Reuse hidden buffer (hiddenSize)
        
        NetMath.scalarSubtract(1.0f, updateGate, oneMinusUpdate);        // (1 - z_t)
        NetMath.elementwiseMultiply(oneMinusUpdate, prevHidden, term1);  // (1 - z_t) ⊙ h_{t-1}
        NetMath.elementwiseMultiply(updateGate, candidate, term2);       // z_t ⊙ h̃_t
        NetMath.elementwiseAdd(term1, term2, newHidden);                 // Final sum
        
        // Restore original concatenatedInput for BPTT (before candidate computation)
        System.arraycopy(prevHidden, 0, concatenatedInput, inputSize, hiddenSize);
    }
    
    
    /**
     * Parallel forward pass through GRU cell using ExecutorService for gate computations.
     * Parallelizes the three gate computations when hidden size is large enough.
     * Optionally applies layer normalization to gate pre-activations.
     */
    private void forwardCellParallel(float[] input, float[] prevHidden, float[] newHidden,
                                   float[] resetGate, float[] updateGate, float[] candidate,
                                   float[] concatenatedInput, GruBuffers buffers, ExecutorService executor) {
        // Concatenate input and previous hidden state
        System.arraycopy(input, 0, concatenatedInput, 0, inputSize);
        System.arraycopy(prevHidden, 0, concatenatedInput, inputSize, hiddenSize);
        
        try {
            // Parallelize the three gate computations
            var resetFuture = executor.submit(() -> {
                NetMath.matrixPreActivationsColumnMajor(concatenatedInput, resetWeights, resetBias, resetGate);
                if (useLayerNorm && resetLayerNorm != null) {
                    resetLayerNorm.forward(resetGate, resetGate);
                }
                SigmoidActivator.INSTANCE.activate(resetGate, resetGate);
                return null;
            });
            
            var updateFuture = executor.submit(() -> {
                NetMath.matrixPreActivationsColumnMajor(concatenatedInput, updateWeights, updateBias, updateGate);
                if (useLayerNorm && updateLayerNorm != null) {
                    updateLayerNorm.forward(updateGate, updateGate);
                }
                SigmoidActivator.INSTANCE.activate(updateGate, updateGate);
                return null;
            });
            
            // Wait for reset and update gates to complete
            resetFuture.get();
            updateFuture.get();
            
            // Compute candidate gate (depends on reset gate)
            float[] resetHiddenProduct = buffers.tempBuffer1;
            NetMath.elementwiseMultiply(resetGate, prevHidden, resetHiddenProduct);
            System.arraycopy(resetHiddenProduct, 0, concatenatedInput, inputSize, hiddenSize);
            
            NetMath.matrixPreActivationsColumnMajor(concatenatedInput, candidateWeights, candidateBias, candidate);
            if (useLayerNorm && candidateLayerNorm != null) {
                candidateLayerNorm.forward(candidate, candidate);
            }
            TanhActivator.INSTANCE.activate(candidate, candidate);
            
            // Final hidden state computation
            float[] oneMinusUpdate = buffers.tempBuffer1;
            float[] term1 = buffers.tempBuffer2;
            float[] term2 = buffers.hiddenStateBuffer;
            
            NetMath.scalarSubtract(1.0f, updateGate, oneMinusUpdate);
            NetMath.elementwiseMultiply(oneMinusUpdate, prevHidden, term1);
            NetMath.elementwiseMultiply(updateGate, candidate, term2);
            NetMath.elementwiseAdd(term1, term2, newHidden);
            
            // Restore original concatenatedInput for BPTT
            System.arraycopy(prevHidden, 0, concatenatedInput, inputSize, hiddenSize);
        } catch (Exception e) {
            throw new RuntimeException("Parallel GRU forward pass failed", e);
        }
    }
    
    @Override
    public LayerContext forward(float[] input, ExecutorService executor) {
        validateInputs(input);
        int seqLen = input.length / inputSize;
        
        boolean useParallel = shouldUseParallel(executor, seqLen);
        if (!useParallel) {
            // Use sequential implementation for small work sizes or no executor
            return forward(input);
        }
        
        // For large work sizes, parallelize gate computations within each timestep
        if (seqLen == 0)
            throw new IllegalArgumentException("Sequence length cannot be zero");
        
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
        java.util.Arrays.fill(hiddenStates[0], 0.0f);
        
        // Process sequence timestep by timestep (cannot parallelize across timesteps)
        for (int t = 0; t < seqLen; t++) {
            // Extract current timestep input
            System.arraycopy(input, t * inputSize, buffers.currentInputBuffer, 0, inputSize);
            
            // Forward through GRU cell with executor-aware implementation
            forwardCellParallel(buffers.currentInputBuffer, hiddenStates[t], hiddenStates[t + 1], 
                              resetGates[t], updateGates[t], candidates[t], concatenatedInputs[t], buffers, executor);
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
        
        // Copy data from ThreadLocal buffers to fresh arrays
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(resetGates[t], 0, freshResetGates[t], 0, hiddenSize);
            System.arraycopy(updateGates[t], 0, freshUpdateGates[t], 0, hiddenSize);
            System.arraycopy(candidates[t], 0, freshCandidates[t], 0, hiddenSize);
            System.arraycopy(hiddenStates[t + 1], 0, freshHiddenStates[t + 1], 0, hiddenSize);
            System.arraycopy(concatenatedInputs[t], 0, freshConcatenatedInputs[t], 0, totalInputSize);
        }
        // Copy initial hidden state h_0
        System.arraycopy(hiddenStates[0], 0, freshHiddenStates[0], 0, hiddenSize);
        
        // Use safe factory method for ThreadLocal buffers - creates a custom context with input copy
        return new GruLayerContext(input.clone(), null, outputs, seqLen, 
                                 freshResetGates, freshUpdateGates, freshCandidates, freshHiddenStates, freshConcatenatedInputs);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient, ExecutorService executor) {
        validateBackwardInputs(stack, stackIndex, upstreamGradient);
        
        LayerContext context = stack[stackIndex];
        GruLayerContext gruContext = (GruLayerContext) context;
        
        int seqLen = gruContext.seqLen;
        float[] inputGradients = new float[gruContext.inputs().length]; // [seqLen * inputSize]
        
        // Get consolidated buffers for this thread
        GruBuffers buffers = allBuffers.get();
        
        // Use consolidated weight gradient accumulators to avoid allocation
        float[][] resetWeightGrads = buffers.resetWeightGradients;
        float[][] updateWeightGrads = buffers.updateWeightGradients;
        float[][] candidateWeightGrads = buffers.candidateWeightGradients;
        
        // Clear weight gradient accumulators
        clearWeightGradients(resetWeightGrads);
        clearWeightGradients(updateWeightGrads);
        clearWeightGradients(candidateWeightGrads);
        
        // Use consolidated buffers for bias gradients  
        float[] resetBiasGrads = buffers.resetGateBuffer;      // Reuse (hiddenSize)
        float[] updateBiasGrads = buffers.updateGateBuffer;    // Reuse (hiddenSize)
        float[] candidateBiasGrads = buffers.candidateBuffer;  // Reuse (hiddenSize)
        
        // Clear bias gradient accumulators
        java.util.Arrays.fill(resetBiasGrads, 0.0f);
        java.util.Arrays.fill(updateBiasGrads, 0.0f);
        java.util.Arrays.fill(candidateBiasGrads, 0.0f);
        
        // Initialize hidden state gradient for BPTT using consolidated buffer
        float[] hiddenGradient = buffers.hiddenStateBuffer;   // Reuse (hiddenSize)
        java.util.Arrays.fill(hiddenGradient, 0.0f);
        
        // Backward pass through time
        for (int t = seqLen - 1; t >= 0; t--) {
            // Add upstream gradient for this timestep to hidden gradient
            if (outputMode == OutputMode.ALL_TIMESTEPS) {
                // For ALL_TIMESTEPS mode, upstream gradient contains gradients for all timesteps
                NetMath.elementwiseAdd(hiddenGradient, 
                                     java.util.Arrays.copyOfRange(upstreamGradient, t * hiddenSize, (t + 1) * hiddenSize), 
                                     hiddenGradient);
            } else if (outputMode == OutputMode.LAST_TIMESTEP && t == seqLen - 1) {
                // For LAST_TIMESTEP mode, upstream gradient only contains gradient for last timestep
                NetMath.elementwiseAdd(hiddenGradient, upstreamGradient, hiddenGradient);
            }
            
            // Compute gradients for this timestep with optimized implementation
            backwardTimestepOptimized(t, gruContext, hiddenGradient,
                                    resetWeightGrads, updateWeightGrads, candidateWeightGrads,
                                    resetBiasGrads, updateBiasGrads, candidateBiasGrads,
                                    inputGradients, buffers, executor);
        }
        
        // Update parameters using accumulated gradients with executor support
        optimizer.optimize(resetWeights, resetBias, resetWeightGrads, resetBiasGrads, executor);
        optimizer.optimize(updateWeights, updateBias, updateWeightGrads, updateBiasGrads, executor);
        optimizer.optimize(candidateWeights, candidateBias, candidateWeightGrads, candidateBiasGrads, executor);
        
        return inputGradients;
    }
    
    /**
     * Clear weight gradient matrix efficiently.
     */
    private void clearWeightGradients(float[][] gradients) {
        for (int i = 0; i < gradients.length; i++) {
            java.util.Arrays.fill(gradients[i], 0.0f);
        }
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
        return backward(stack, stackIndex, upstreamGradient, (ExecutorService) null);
    }
    
    /**
     * Optimized computation of gradients for a single timestep in BPTT.
     * Uses vectorized operations with consolidated buffers to eliminate allocations.
     */
    private void backwardTimestepOptimized(int t, GruLayerContext context, float[] hiddenGradient,
                                         float[][] resetWeightGrads, float[][] updateWeightGrads, float[][] candidateWeightGrads,
                                         float[] resetBiasGrads, float[] updateBiasGrads, float[] candidateBiasGrads,
                                         float[] inputGradients, GruBuffers buffers, ExecutorService executor) {
        
        // Get stored values from forward pass
        float[] resetGate = context.resetGates[t];
        float[] updateGate = context.updateGates[t];
        float[] candidate = context.candidates[t];
        float[] prevHidden = context.hiddenStates[t];     // h_{t-1}
        float[] concatenatedInput = context.concatenatedInputs[t];
        
        // Get current timestep input using consolidated buffer to avoid allocation
        float[] currentInput = buffers.currentInputBuffer;
        validateBuffer(currentInput, "currentInputBuffers");
        System.arraycopy(context.inputs(), t * inputSize, currentInput, 0, inputSize);
        
        // === Gradient w.r.t. final hidden state computation === 
        // h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        
        // Use consolidated buffers for hiddenSize arrays (all same size) - ZERO ALLOCATION
        float[] candidateMinusPrev = buffers.tempBuffer1;       // hiddenSize - reuse temp buffer
        float[] updateGateGrad = buffers.tempBuffer2;           // hiddenSize - reuse temp buffer
        float[] candidateGrad = buffers.resetGateBuffer;        // hiddenSize - reuse gate buffer
        
        // Gradient w.r.t. update gate: ∂L/∂z_t = ∂L/∂h_t ⊙ (h̃_t - h_{t-1})
        NetMath.elementwiseSubtract(candidate, prevHidden, candidateMinusPrev);
        NetMath.elementwiseMultiply(hiddenGradient, candidateMinusPrev, updateGateGrad);
        
        // Gradient w.r.t. candidate: ∂L/∂h̃_t = ∂L/∂h_t ⊙ z_t
        NetMath.elementwiseMultiply(hiddenGradient, updateGate, candidateGrad);
        
        // Gradient w.r.t. previous hidden state from final computation - use consolidated buffers
        float[] oneMinusUpdate = buffers.updateGateBuffer;      // hiddenSize - reuse gate buffer
        float[] prevHiddenGradFromFinal = candidateMinusPrev;    // Reuse: hiddenSize
        NetMath.scalarSubtract(1.0f, updateGate, oneMinusUpdate);
        NetMath.elementwiseMultiply(hiddenGradient, oneMinusUpdate, prevHiddenGradFromFinal);
        
        // === Gradient through candidate computation ===
        // h̃_t = tanh(W_h * [r_t ⊙ h_{t-1}, x_t] + b_h)
        
        // Apply tanh derivative: ∂tanh/∂x = 1 - tanh²(x) - reuse candidateGrad buffer
        float[] candidatePreGrad = candidateGrad; // Reuse: hiddenSize
        TanhActivator.INSTANCE.derivative(candidate, candidatePreGrad, executor);
        NetMath.elementwiseMultiply(candidateGrad, candidatePreGrad, candidatePreGrad); // Chain rule
        
        // Accumulate candidate bias gradients
        NetMath.elementwiseAdd(candidateBiasGrads, candidatePreGrad, candidateBiasGrads);
        
        // Parallel computation of heavy matrix operations when conditions are met
        if (shouldUseParallelForTimestep(executor)) {
            try {
                // Parallelize the three heavy matrix operations
                var candidateWeightFuture = executor.submit(() -> {
                    NetMath.matrixOuterProduct(concatenatedInput, candidatePreGrad, candidateWeightGrads);
                    return null;
                });
                
                var candidateConcatFuture = executor.submit(() -> {
                    float[] candidateConcatGrad = buffers.concatenatedInputBuffer;
                    NetMath.matrixVectorMultiplyColumnMajor(candidateWeights, candidatePreGrad, candidateConcatGrad);
                    return candidateConcatGrad;
                });
                
                // Wait for completion
                candidateWeightFuture.get();
                float[] candidateConcatGrad = candidateConcatFuture.get();
                
            } catch (Exception e) {
                throw new RuntimeException("Parallel backward pass failed", e);
            }
        } else {
            // Sequential computation for small layers or no executor
            NetMath.matrixOuterProduct(concatenatedInput, candidatePreGrad, candidateWeightGrads);
            float[] candidateConcatGrad = buffers.concatenatedInputBuffer;
            NetMath.matrixVectorMultiplyColumnMajor(candidateWeights, candidatePreGrad, candidateConcatGrad);
        }
        
        // Get the concatenated gradient buffer for subsequent operations
        float[] candidateConcatGrad = buffers.concatenatedInputBuffer;
        
        // Extract gradients - use existing buffer for input gradients
        float[] inputGradFromCandidate = currentInput; // Reuse: inputSize
        float[] resetHiddenProductGrad = updateGateGrad; // Reuse: hiddenSize
        System.arraycopy(candidateConcatGrad, 0, inputGradFromCandidate, 0, inputSize);
        System.arraycopy(candidateConcatGrad, inputSize, resetHiddenProductGrad, 0, hiddenSize);
        
        // === Gradient through reset gate ===
        // r_t ⊙ h_{t-1} → gradients w.r.t. r_t and h_{t-1}
        
        // Gradient w.r.t. reset gate: ∂L/∂r_t = ∂L/∂(r_t ⊙ h_{t-1}) ⊙ h_{t-1}
        float[] resetGateGrad = oneMinusUpdate; // Reuse: hiddenSize
        NetMath.elementwiseMultiply(resetHiddenProductGrad, prevHidden, resetGateGrad);
        
        // Gradient w.r.t. previous hidden state from reset operation
        float[] prevHiddenGradFromReset = resetHiddenProductGrad; // Reuse in-place: hiddenSize
        NetMath.elementwiseMultiply(resetHiddenProductGrad, resetGate, prevHiddenGradFromReset);
        
        // === Gradient through reset gate computation ===
        // r_t = σ(W_r * [h_{t-1}, x_t] + b_r)
        
        // Apply sigmoid derivative: ∂σ/∂x = σ(x) * (1 - σ(x)) - reuse resetGateGrad buffer
        float[] resetPreGrad = resetGateGrad; // Reuse: hiddenSize
        SigmoidActivator.INSTANCE.derivative(resetGate, resetPreGrad, executor);
        NetMath.elementwiseMultiply(resetGateGrad, resetPreGrad, resetPreGrad); // Chain rule
        
        // Accumulate reset bias gradients
        NetMath.elementwiseAdd(resetBiasGrads, resetPreGrad, resetBiasGrads);
        
        // Parallel computation for reset gate when conditions are met
        if (shouldUseParallelForTimestep(executor)) {
            try {
                var resetWeightFuture = executor.submit(() -> {
                    NetMath.matrixOuterProduct(concatenatedInput, resetPreGrad, resetWeightGrads);
                    return null;
                });
                
                var resetConcatFuture = executor.submit(() -> {
                    float[] resetConcatGrad = candidateConcatGrad; // Reuse: inputSize + hiddenSize
                    NetMath.matrixVectorMultiplyColumnMajor(resetWeights, resetPreGrad, resetConcatGrad);
                    return null;
                });
                
                resetWeightFuture.get();
                resetConcatFuture.get();
                
            } catch (Exception e) {
                throw new RuntimeException("Parallel backward pass failed", e);
            }
        } else {
            // Sequential computation
            NetMath.matrixOuterProduct(concatenatedInput, resetPreGrad, resetWeightGrads);
            NetMath.matrixVectorMultiplyColumnMajor(resetWeights, resetPreGrad, candidateConcatGrad);
        }
        
        float[] resetConcatGrad = candidateConcatGrad; // Reference for subsequent operations
        
        // === Gradient through update gate computation ===
        // z_t = σ(W_z * [h_{t-1}, x_t] + b_z)
        
        // Apply sigmoid derivative - reuse candidatePreGrad buffer (no longer needed)
        float[] updatePreGrad = candidatePreGrad; // Reuse: hiddenSize
        SigmoidActivator.INSTANCE.derivative(updateGate, updatePreGrad, executor);
        NetMath.elementwiseMultiply(updateGateGrad, updatePreGrad, updatePreGrad); // Chain rule
        
        // Accumulate update bias gradients
        NetMath.elementwiseAdd(updateBiasGrads, updatePreGrad, updateBiasGrads);
        
        // Parallel computation for update gate when conditions are met
        if (shouldUseParallelForTimestep(executor)) {
            try {
                var updateWeightFuture = executor.submit(() -> {
                    NetMath.matrixOuterProduct(concatenatedInput, updatePreGrad, updateWeightGrads);
                    return null;
                });
                
                var updateConcatFuture = executor.submit(() -> {
                    float[] updateConcatGrad = buffers.tempConcatBuffer;
                    NetMath.matrixVectorMultiplyColumnMajor(updateWeights, updatePreGrad, updateConcatGrad);
                    return null;
                });
                
                updateWeightFuture.get();
                updateConcatFuture.get();
                
            } catch (Exception e) {
                throw new RuntimeException("Parallel backward pass failed", e);
            }
        } else {
            // Sequential computation
            NetMath.matrixOuterProduct(concatenatedInput, updatePreGrad, updateWeightGrads);
            NetMath.matrixVectorMultiplyColumnMajor(updateWeights, updatePreGrad, buffers.tempConcatBuffer);
        }
        
        float[] updateConcatGrad = buffers.tempConcatBuffer; // Reference for subsequent operations
        
        // === Accumulate gradients w.r.t. inputs and previous hidden state ===
        
        // Input gradients: sum from all three gates - ZERO ALLOCATION approach
        float[] totalInputGrad = inputGradFromCandidate; // Already set above, reuse: inputSize
        
        // Add input gradients from reset gate directly (no intermediate allocation)
        for (int i = 0; i < inputSize; i++) {
            totalInputGrad[i] += resetConcatGrad[i];
        }
        
        // Add input gradients from update gate directly (no intermediate allocation)
        for (int i = 0; i < inputSize; i++) {
            totalInputGrad[i] += updateConcatGrad[i];
        }
        
        // Store input gradient for this timestep
        System.arraycopy(totalInputGrad, 0, inputGradients, t * inputSize, inputSize);
        
        // Previous hidden state gradients: sum from all sources - ZERO ALLOCATION approach
        if (t > 0) { // Don't propagate to h_{-1} (initial state)  
            // Start with gradient from final hidden state computation - reuse prevHiddenGradFromFinal
            float[] totalHiddenGrad = prevHiddenGradFromFinal; // Already set above, reuse: hiddenSize
            
            // Add gradient from reset operation
            NetMath.elementwiseAdd(totalHiddenGrad, prevHiddenGradFromReset, totalHiddenGrad);
            
            // Add hidden gradients from reset gate computation directly (no intermediate allocation)
            for (int i = 0; i < hiddenSize; i++) {
                totalHiddenGrad[i] += resetConcatGrad[inputSize + i];
            }
            
            // Add hidden gradients from update gate computation directly (no intermediate allocation)
            for (int i = 0; i < hiddenSize; i++) {
                totalHiddenGrad[i] += updateConcatGrad[inputSize + i];
            }
            
            // Set hidden gradient for next (previous) timestep
            System.arraycopy(totalHiddenGrad, 0, hiddenGradient, 0, hiddenSize);
        } else {
            // Clear hidden gradient for t=0 (no previous timestep)
            java.util.Arrays.fill(hiddenGradient, 0.0f);
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
        public Layer create(int inputSize) {
            return createLayer(inputSize, getEffectiveOptimizer(null));
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer) {
            return new GruLayer(effectiveOptimizer, hiddenSize, inputSize, initStrategy);
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
        public Layer create(Shape inputShape, Optimizer effectiveOptimizer) {
            if (effectiveOptimizer == null) {
                effectiveOptimizer = getEffectiveOptimizer(null);
            }
            
            if (inputShape.rank() == 2) {
                // Perfect! We have [seqLen, features]
                int features = inputShape.dim(1);
                return new GruLayer(effectiveOptimizer, hiddenSize, features, initStrategy, OutputMode.ALL_TIMESTEPS);
            } else if (inputShape.rank() == 1) {
                // Fall back to old behavior
                return create(inputShape.toFlatSize(), effectiveOptimizer);
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
        public Layer create(int inputSize) {
            // For GRU, inputSize = sequenceLength × inputDimension
            // This is set by the previous layer (e.g., InputEmbeddingLayer outputs seqLen × embeddingDim)
            return createLayer(inputSize, getEffectiveOptimizer(null));
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer) {
            // For GRU, inputSize is usually seqLen * embeddingDim from InputSequenceEmbeddingLayer
            // GRU needs per-timestep size (embeddingDim), not the full flattened size
            
            int perTimestepSize;
            
            // Use expectedInputDimension if provided
            if (expectedInputDimension > 0 && inputSize % expectedInputDimension == 0) {
                perTimestepSize = expectedInputDimension;
            } else if (cachedInputDimension > 0 && inputSize % cachedInputDimension == 0) {
                // Use cached dimension if available
                perTimestepSize = cachedInputDimension;
            } else {
                // Cannot determine per-timestep size - this should be resolved through Shape API
                throw new IllegalArgumentException(
                    "Cannot determine per-timestep input size for GRU. Total input size: " + inputSize +
                    ". Please use GruLayer.specAll(hiddenSize, optimizer, initStrategy, expectedInputDimension) " +
                    "or use the Shape API to provide dimension information.");
            }
            
            // Create the GRU layer with per-timestep input size
            GruLayer layer = new GruLayer(effectiveOptimizer, hiddenSize, perTimestepSize, initStrategy, OutputMode.ALL_TIMESTEPS);
            // Cache the input dimension for output size calculation
            this.cachedInputDimension = layer.inputSize;
            return layer;
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
        public Layer create(int inputSize) {
            return createLayer(inputSize, getEffectiveOptimizer(null));
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer) {
            // For GRU, inputSize is usually seqLen * embeddingDim from InputSequenceEmbeddingLayer
            // GRU needs per-timestep size (embeddingDim), not the full flattened size
            
            // TODO: This is a temporary solution until Shape API is implemented
            // Try to infer the per-timestep size based on common embedding dimensions
            int perTimestepSize = inputSize; // Default to full size if single timestep
            
            // Check if this is likely a sequence
            if (inputSize > 128) {  // Lower threshold to catch more cases
                // Check common embedding dimensions
                int[] commonEmbedDims = {128, 256, 64, 512, 768, 1024, 300, 100, 50};
                for (int embDim : commonEmbedDims) {
                    if (inputSize % embDim == 0) {
                        int seqLen = inputSize / embDim;
                        // Validate reasonable sequence length
                        if (seqLen >= 5 && seqLen <= 500) {
                            perTimestepSize = embDim;
                            break;
                        }
                    }
                }
            }
            
            return new GruLayer(effectiveOptimizer, hiddenSize, perTimestepSize, initStrategy, OutputMode.LAST_TIMESTEP);
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
    
    public static GruLayer deserialize(DataInputStream in, int version) throws IOException {
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
        
        // Create layer with correct output mode
        GruLayer layer = new GruLayer(optimizer, hiddenSize, inputSize, WeightInitStrategy.XAVIER, outputMode);
        
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
                        LayerContext context = forward(sequenceInput);
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
                
                LayerContext context = forward(sequenceInput);
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
        public Layer create(int inputSize) {
            return create(inputSize, null);
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer) {
            // For language models, inputSize is usually seqLen * embeddingDim from InputSequenceEmbeddingLayer
            // GRU needs per-timestep size (embeddingDim), not the full flattened size
            // We'll use the same logic as the regular GRU specs to infer the per-timestep size
            
            // Check common embedding dimensions in order of likelihood
            int[] commonEmbedDims = {128, 256, 64, 512, 768, 1024, 300, 100, 50};
            int perTimestepSize = inputSize; // Default to full size if we can't infer
            
            for (int embDim : commonEmbedDims) {
                if (inputSize % embDim == 0) {
                    int seqLen = inputSize / embDim;
                    // Check if this gives a reasonable sequence length
                    if (seqLen >= 5 && seqLen <= 500) {
                        perTimestepSize = embDim;
                        break;
                    }
                }
            }
            
            return new GruLayer(effectiveOptimizer, hiddenSize, perTimestepSize, initStrategy, OutputMode.ALL_TIMESTEPS, true);
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
        public Layer create(int inputSize) {
            return create(inputSize, null);
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer) {
            // For language models, inputSize is usually seqLen * embeddingDim from InputSequenceEmbeddingLayer
            // GRU needs per-timestep size (embeddingDim), not the full flattened size
            // We'll use the same logic as the regular GRU specs to infer the per-timestep size
            
            // Check common embedding dimensions in order of likelihood
            int[] commonEmbedDims = {128, 256, 64, 512, 768, 1024, 300, 100, 50};
            int perTimestepSize = inputSize; // Default to full size if we can't infer
            
            for (int embDim : commonEmbedDims) {
                if (inputSize % embDim == 0) {
                    int seqLen = inputSize / embDim;
                    // Check if this gives a reasonable sequence length
                    if (seqLen >= 5 && seqLen <= 500) {
                        perTimestepSize = embDim;
                        break;
                    }
                }
            }
            
            return new GruLayer(effectiveOptimizer, hiddenSize, perTimestepSize, initStrategy, OutputMode.LAST_TIMESTEP, true);
        }
    }
}