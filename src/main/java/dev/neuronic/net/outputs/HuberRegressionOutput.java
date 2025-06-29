package dev.neuronic.net.outputs;

import dev.neuronic.net.layers.BaseLayerSpec;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.GradientAccumulator;
import dev.neuronic.net.common.PooledFloatArray;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.losses.HuberLoss;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.serialization.SerializationService;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Linear output layer with Huber loss for robust regression.
 * 
 * <p>Computes: output = Wx + b (no activation)
 * <p>Uses Huber loss which is less sensitive to outliers than MSE.
 * 
 * <p>Use for:
 * <ul>
 *   <li>Regression tasks with potential outliers in the data
 *   <li>When you want a smoother alternative to MAE
 *   <li>Robust parameter estimation
 *   <li>Any regression where extreme errors should have limited influence
 * </ul>
 * 
 * <p>The delta parameter controls the transition from quadratic to linear loss:
 * <ul>
 *   <li>Small delta (0.5): More MSE-like, sensitive to small errors
 *   <li>Default delta (1.0): Balanced behavior
 *   <li>Large delta (2.0+): More MAE-like, robust to outliers
 * </ul>
 */
public class HuberRegressionOutput implements Layer, GradientAccumulator, Serializable, RegressionOutput {
    
    private final Optimizer optimizer;
    private final float[][] weights;
    private final float[] biases;
    private final int outputs;
    private final int inputs;
    private final float delta;
    private final HuberLoss huberLoss;
    // Instance buffer pools for different array sizes
    private final PooledFloatArray outputBufferPool;
    private final PooledFloatArray gradientBufferPool;
    private final PooledFloatArray inputBufferPool;
    
    // Gradient accumulation state
    private final ThreadLocal<float[][]> accumulatedWeightGradients;
    private final ThreadLocal<float[]> accumulatedBiasGradients;
    private final ThreadLocal<Boolean> accumulating;
    
    public HuberRegressionOutput(Optimizer optimizer, int outputs, int inputs, float delta) {
        this.optimizer = optimizer;
        this.weights = new float[inputs][outputs];
        this.biases = new float[outputs];
        this.outputs = outputs;
        this.inputs = inputs;
        this.delta = delta;
        this.huberLoss = HuberLoss.create(delta);
        // Initialize buffer pools
        this.outputBufferPool = new PooledFloatArray(outputs);
        this.gradientBufferPool = new PooledFloatArray(outputs);
        this.inputBufferPool = new PooledFloatArray(inputs);
        
        // Initialize gradient accumulation state
        this.accumulatedWeightGradients = ThreadLocal.withInitial(() -> new float[inputs][outputs]);
        this.accumulatedBiasGradients = ThreadLocal.withInitial(() -> new float[outputs]);
        this.accumulating = ThreadLocal.withInitial(() -> Boolean.FALSE);
        
        // Xavier initialization for linear layers
        NetMath.weightInitXavier(weights, inputs, outputs);
        NetMath.biasInit(biases, 0.0f);
    }
    
    @Override
    public LayerContext forward(float[] input) {
        // Allocate new array for LayerContext - never use ThreadLocal buffers in contexts
        float[] output = new float[outputs];
        
        // Linear transformation: output = Wx + b
        System.arraycopy(biases, 0, output, 0, outputs);
        for (int inputIdx = 0; inputIdx < inputs; inputIdx++) {
            float inputValue = input[inputIdx];
            for (int outputIdx = 0; outputIdx < outputs; outputIdx++)
                output[outputIdx] += inputValue * weights[inputIdx][outputIdx];
        }
        
        return new LayerContext(input, output, output); // preActivations == outputs for linear
    }
    
    /**
     * Compute Huber loss.
     */
    public float computeLoss(float[] predictions, float[] targets) {
        return huberLoss.loss(predictions, targets);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] targets) {
        LayerContext context = stack[stackIndex];
        
        float[] gradients = gradientBufferPool.getBuffer();
        float[] downstreamGradient = inputBufferPool.getBuffer();
        
        try {
            // Huber loss gradient
            NetMath.lossDerivativesHuber(context.outputs(), targets, delta, gradients);
            
            // Compute weight gradients using NetMath
            float[][] weightGradients = new float[inputs][outputs];
            NetMath.matrixWeightGradientsColumnMajor(context.inputs(), gradients, weightGradients);
            
            // Update weights and biases
            optimizer.optimize(weights, biases, weightGradients, gradients);
            
            // Compute downstream gradient using NetMath
            NetMath.matrixVectorMultiplyColumnMajor(weights, gradients, downstreamGradient);
            
            // Return a fresh copy
            float[] result = new float[inputs];
            System.arraycopy(downstreamGradient, 0, result, 0, inputs);
            return result;
            
        } finally {
            gradientBufferPool.releaseBuffer(gradients);
            inputBufferPool.releaseBuffer(downstreamGradient);
        }
    }
    
    @Override
    public int getOutputSize() {
        return outputs;
    }
    
    // Gradient accumulation implementation
    
    @Override
    public void startAccumulation() {
        accumulating.set(Boolean.TRUE);
        // Zero out accumulated gradients
        float[][] weightGrads = accumulatedWeightGradients.get();
        float[] biasGrads = accumulatedBiasGradients.get();
        
        NetMath.matrixInit(weightGrads, 0.0f);
        NetMath.biasInit(biasGrads, 0.0f);
    }
    
    @Override
    public float[] backwardAccumulate(LayerContext[] stack, int stackIndex, float[] targets) {
        LayerContext context = stack[stackIndex];
        
        float[] gradients = gradientBufferPool.getBuffer();
        
        try {
            // Huber loss gradient
            NetMath.lossDerivativesHuber(context.outputs(), targets, delta, gradients);
            
            // Compute weight gradients using NetMath
            float[][] weightGradients = new float[inputs][outputs];
            NetMath.matrixWeightGradientsColumnMajor(context.inputs(), gradients, weightGradients);
            
            // Accumulate gradients
            float[][] accWeightGrads = accumulatedWeightGradients.get();
            float[] accBiasGrads = accumulatedBiasGradients.get();
            
            // Add weight gradients to accumulated
            for (int i = 0; i < inputs; i++)
                NetMath.elementwiseAdd(accWeightGrads[i], weightGradients[i], accWeightGrads[i]);
            
            // Add bias gradients to accumulated
            NetMath.elementwiseAdd(accBiasGrads, gradients, accBiasGrads);
            
            // Compute downstream gradient using NetMath
            float[] downstreamGradient = inputBufferPool.getBuffer();
            try {
                NetMath.matrixVectorMultiplyColumnMajor(weights, gradients, downstreamGradient);
                
                // Return a fresh copy
                float[] result = new float[inputs];
                System.arraycopy(downstreamGradient, 0, result, 0, inputs);
                return result;
            } finally {
                inputBufferPool.releaseBuffer(downstreamGradient);
            }
            
        } finally {
            gradientBufferPool.releaseBuffer(gradients);
        }
    }
    
    @Override
    public void applyAccumulatedGradients(int batchSize) {
        if (!accumulating.get())
            return;
        
        float[][] accWeightGrads = accumulatedWeightGradients.get();
        float[] accBiasGrads = accumulatedBiasGradients.get();
        
        // Average the accumulated gradients
        float scale = 1.0f / batchSize;
        for (int i = 0; i < inputs; i++)
            NetMath.elementwiseScaleInPlace(accWeightGrads[i], scale);
        
        NetMath.elementwiseScaleInPlace(accBiasGrads, scale);
        
        // Update weights and biases
        optimizer.optimize(weights, biases, accWeightGrads, accBiasGrads);
        
        accumulating.set(Boolean.FALSE);
    }
    
    @Override
    public boolean isAccumulating() {
        return accumulating.get();
    }
    
    /**
     * Create a Huber regression output with default delta (1.0).
     */
    public static Layer.Spec spec(int outputs, Optimizer optimizer) {
        return new HuberRegressionOutputSpec(outputs, optimizer, 1.0f, 1.0);
    }
    
    /**
     * Create a Huber regression output with custom delta.
     * 
     * @param outputs number of output values
     * @param optimizer optimizer for this layer
     * @param delta Huber loss threshold parameter
     */
    public static Layer.Spec spec(int outputs, Optimizer optimizer, float delta) {
        return new HuberRegressionOutputSpec(outputs, optimizer, delta, 1.0);
    }
    
    /**
     * Create a Huber regression output with custom delta and learning rate ratio.
     * 
     * @param outputs number of output values
     * @param optimizer optimizer for this layer (null to use default)
     * @param delta Huber loss threshold parameter
     * @param learningRateRatio learning rate scaling factor (1.0 = normal)
     */
    public static Layer.Spec spec(int outputs, Optimizer optimizer, float delta, double learningRateRatio) {
        return new HuberRegressionOutputSpec(outputs, optimizer, delta, learningRateRatio);
    }
    
    /**
     * Specification for creating Huber regression output layers with optimizer management.
     */
    private static class HuberRegressionOutputSpec extends BaseLayerSpec<HuberRegressionOutputSpec> {
        private final int outputs;
        private final float delta;
        
        public HuberRegressionOutputSpec(int outputs, Optimizer optimizer, float delta, double learningRateRatio) {
            super(outputs, optimizer);
            this.outputs = outputs;
            this.delta = delta;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        @Override
        public Layer create(int inputSize) {
            return createLayer(inputSize, getEffectiveOptimizer(null));
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer) {
            return new HuberRegressionOutput(effectiveOptimizer, outputs, inputSize, delta);
        }
        
        @Override
        public int getOutputSize() {
            return outputs;
        }
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        // Write layer dimensions
        out.writeInt(outputs);
        out.writeInt(inputs);
        out.writeFloat(delta);
        
        // Write weights (column-major format)
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < outputs; j++)
                out.writeFloat(weights[i][j]);
        }
        
        // Write biases
        for (float bias : biases)
            out.writeFloat(bias);
        
        // Write optimizer
        Serializable serializableOptimizer = (Serializable) optimizer;
        out.writeInt(serializableOptimizer.getTypeId());
        serializableOptimizer.writeTo(out, version);
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize(DataInputStream, int) static method instead");
    }
    
    /**
     * Static method to deserialize a HuberRegressionOutput from stream.
     */
    public static HuberRegressionOutput deserialize(DataInputStream in, int version) throws IOException {
        // Read layer dimensions
        int outputs = in.readInt();
        int inputs = in.readInt();
        float delta = in.readFloat();
        
        // Read weights
        float[][] weights = new float[inputs][outputs];
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < outputs; j++)
                weights[i][j] = in.readFloat();
        }
        
        // Read biases
        float[] biases = new float[outputs];
        for (int i = 0; i < outputs; i++)
            biases[i] = in.readFloat();
        
        // Read optimizer using centralized service
        int optimizerTypeId = in.readInt();
        Optimizer optimizer = SerializationService.deserializeOptimizer(in, optimizerTypeId, version);
        
        // Create layer and set weights
        HuberRegressionOutput layer = new HuberRegressionOutput(optimizer, outputs, inputs, delta);
        
        // Copy weights and biases
        for (int i = 0; i < inputs; i++)
            System.arraycopy(weights[i], 0, layer.weights[i], 0, outputs);
        
        System.arraycopy(biases, 0, layer.biases, 0, outputs);
        
        return layer;
    }
    
    @Override
    public int getSerializedSize(int version) {
        int size = 12; // outputs + inputs + delta
        size += inputs * outputs * 4; // weights
        size += outputs * 4; // biases
        size += 4; // optimizer type ID
        size += ((Serializable) optimizer).getSerializedSize(version); // optimizer data
        return size;
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_HUBER_REGRESSION_OUTPUT;
    }
}