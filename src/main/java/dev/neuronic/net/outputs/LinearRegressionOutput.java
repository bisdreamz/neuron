package dev.neuronic.net.outputs;

import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.GradientAccumulator;
import dev.neuronic.net.layers.BaseLayerSpec;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.serialization.SerializationService;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Linear output layer for regression tasks.
 * 
 * Computes: output = Wx + b (no activation)
 * Uses Mean Squared Error loss internally.
 * 
 * Use for:
 * - Continuous value prediction (house prices, stock prices)
 * - Multi-output regression
 * - Any task where outputs are real numbers
 */
public class LinearRegressionOutput implements Layer, GradientAccumulator, Serializable, RegressionOutput {
    
    private final Optimizer optimizer;
    private final float[][] weights;
    private final float[] biases;
    private final int outputs;
    private final int inputs;
    private final ThreadLocal<float[]> outputBuffers;
    private final ThreadLocal<float[]> gradientBuffers;
    private final ThreadLocal<float[]> inputBuffers;
    private final ThreadLocal<float[][]> weightGradientBuffers;
    
    // Gradient accumulation state
    private final ThreadLocal<float[][]> accumulatedWeightGradients;
    private final ThreadLocal<float[]> accumulatedBiasGradients;
    private final ThreadLocal<Boolean> accumulating;
    
    public LinearRegressionOutput(Optimizer optimizer, int outputs, int inputs) {
        this.optimizer = optimizer;
        this.weights = new float[inputs][outputs];
        this.biases = new float[outputs];
        this.outputs = outputs;
        this.inputs = inputs;
        this.outputBuffers = ThreadLocal.withInitial(() -> new float[outputs]);
        this.gradientBuffers = ThreadLocal.withInitial(() -> new float[outputs]);
        this.inputBuffers = ThreadLocal.withInitial(() -> new float[inputs]);
        this.weightGradientBuffers = ThreadLocal.withInitial(() -> new float[inputs][outputs]);
        
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
            for (int outputIdx = 0; outputIdx < outputs; outputIdx++) {
                output[outputIdx] += inputValue * weights[inputIdx][outputIdx];
            }
        }
        
        return new LayerContext(input, output, output); // preActivations == outputs for linear
    }
    
    /**
     * Compute Mean Squared Error loss.
     */
    public float computeLoss(float[] predictions, float[] targets) {
        return NetMath.lossComputeMSE(predictions, targets);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] targets) {
        LayerContext context = stack[stackIndex];
        
        // MSE gradient: 2 * (predictions - targets) / n
        float[] gradients = gradientBuffers.get();
        NetMath.lossDerivativesMSE(context.outputs(), targets, gradients);
        
        // Compute weight gradients using NetMath
        float[][] weightGradients = weightGradientBuffers.get();
        NetMath.matrixWeightGradientsColumnMajor(context.inputs(), gradients, weightGradients);
        
        // Update weights and biases
        optimizer.optimize(weights, biases, weightGradients, gradients);
        
        // Compute downstream gradient using NetMath
        float[] downstreamGradient = inputBuffers.get();
        NetMath.matrixVectorMultiplyColumnMajor(weights, gradients, downstreamGradient);
        
        return downstreamGradient;
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
        
        // MSE gradient
        float[] gradients = gradientBuffers.get();
        NetMath.lossDerivativesMSE(context.outputs(), targets, gradients);
        
        // Compute weight gradients using NetMath
        float[][] weightGradients = weightGradientBuffers.get();
        NetMath.matrixWeightGradientsColumnMajor(context.inputs(), gradients, weightGradients);
        
        // Accumulate gradients
        float[][] accWeightGrads = accumulatedWeightGradients.get();
        float[] accBiasGrads = accumulatedBiasGradients.get();
        
        // Add weight gradients to accumulated
        for (int i = 0; i < inputs; i++) {
            NetMath.elementwiseAdd(accWeightGrads[i], weightGradients[i], accWeightGrads[i]);
        }
        
        // Add bias gradients to accumulated
        NetMath.elementwiseAdd(accBiasGrads, gradients, accBiasGrads);
        
        // Compute downstream gradient using NetMath
        float[] downstreamGradient = inputBuffers.get();
        NetMath.matrixVectorMultiplyColumnMajor(weights, gradients, downstreamGradient);
        
        return downstreamGradient;
    }
    
    @Override
    public void applyAccumulatedGradients(int batchSize) {
        if (!accumulating.get()) return;
        
        float[][] accWeightGrads = accumulatedWeightGradients.get();
        float[] accBiasGrads = accumulatedBiasGradients.get();
        
        // Average the accumulated gradients
        float scale = 1.0f / batchSize;
        for (int i = 0; i < inputs; i++) {
            NetMath.elementwiseScaleInPlace(accWeightGrads[i], scale);
        }
        NetMath.elementwiseScaleInPlace(accBiasGrads, scale);
        
        // Update weights and biases
        optimizer.optimize(weights, biases, accWeightGrads, accBiasGrads);
        
        accumulating.set(Boolean.FALSE);
    }
    
    @Override
    public boolean isAccumulating() {
        return accumulating.get();
    }
    
    public static Layer.Spec spec(int outputs, Optimizer optimizer) {
        return new LinearRegressionOutputSpec(outputs, optimizer, 1.0);
    }
    
    /**
     * Create a linear regression output specification with custom learning rate ratio.
     * 
     * @param outputs number of output values
     * @param optimizer optimizer for this layer (null to use default)
     * @param learningRateRatio learning rate scaling factor (1.0 = normal)
     */
    public static Layer.Spec spec(int outputs, Optimizer optimizer, double learningRateRatio) {
        return new LinearRegressionOutputSpec(outputs, optimizer, learningRateRatio);
    }
    
    /**
     * Specification for creating linear regression output layers with optimizer management.
     */
    private static class LinearRegressionOutputSpec extends BaseLayerSpec<LinearRegressionOutputSpec> {
        private final int outputs;
        
        public LinearRegressionOutputSpec(int outputs, Optimizer optimizer, double learningRateRatio) {
            super(outputs, optimizer);
            this.outputs = outputs;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        @Override
        public Layer create(int inputSize) {
            return createLayer(inputSize, getEffectiveOptimizer(null));
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer) {
            return new LinearRegressionOutput(effectiveOptimizer, outputs, inputSize);
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
        
        // Write weights (column-major format)
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < outputs; j++) {
                out.writeFloat(weights[i][j]);
            }
        }
        
        // Write biases
        for (float bias : biases) {
            out.writeFloat(bias);
        }
        
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
     * Static method to deserialize a LinearRegressionOutput from stream.
     */
    public static LinearRegressionOutput deserialize(DataInputStream in, int version) throws IOException {
        // Read layer dimensions
        int outputs = in.readInt();
        int inputs = in.readInt();
        
        // Read weights
        float[][] weights = new float[inputs][outputs];
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < outputs; j++) {
                weights[i][j] = in.readFloat();
            }
        }
        
        // Read biases
        float[] biases = new float[outputs];
        for (int i = 0; i < outputs; i++) {
            biases[i] = in.readFloat();
        }
        
        // Read optimizer using centralized service
        int optimizerTypeId = in.readInt();
        Optimizer optimizer = SerializationService.deserializeOptimizer(in, optimizerTypeId, version);
        
        // Create layer and set weights
        LinearRegressionOutput layer = new LinearRegressionOutput(optimizer, outputs, inputs);
        
        // Copy weights and biases
        for (int i = 0; i < inputs; i++) {
            System.arraycopy(weights[i], 0, layer.weights[i], 0, outputs);
        }
        System.arraycopy(biases, 0, layer.biases, 0, outputs);
        
        return layer;
    }
    
    @Override
    public int getSerializedSize(int version) {
        int size = 8; // outputs + inputs
        size += inputs * outputs * 4; // weights
        size += outputs * 4; // biases
        size += 4; // optimizer type ID
        size += ((Serializable) optimizer).getSerializedSize(version); // optimizer data
        return size;
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_LINEAR_REGRESSION_OUTPUT;
    }
}