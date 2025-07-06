package dev.neuronic.net.math;

import dev.neuronic.net.math.ops.*;
import dev.neuronic.net.optimizers.adamw.FusedAdamWUpdate;

/**
 * Central entry point for all neural network mathematical operations.
 * Methods are prefixed by operation type for intuitive autocomplete.
 */
public final class NetMath {
    
    // ========== DOT PRODUCT ==========
    
    /**
     * Compute dot product of two arrays.
     */
    public static float dotProduct(float[] a, float[] b) {
        return DotProduct.compute(a, b);
    }
    
    // ========== ELEMENT-WISE OPERATIONS ==========
    
    /**
     * Element-wise multiplication: output[i] = a[i] * b[i]
     */
    public static void elementwiseMultiply(float[] a, float[] b, float[] output) {
        ElementwiseMultiply.compute(a, b, output);
    }
    
    /**
     * Element-wise addition: output[i] = a[i] + b[i]
     */
    public static void elementwiseAdd(float[] a, float[] b, float[] output) {
        ElementwiseAdd.compute(a, b, output);
    }
    
    /**
     * Element-wise subtraction: output[i] = a[i] - b[i]
     */
    public static void elementwiseSubtract(float[] a, float[] b, float[] output) {
        ElementwiseSubtract.compute(a, b, output);
    }
    
    /**
     * Scalar subtraction: output[i] = scalar - array[i]
     */
    public static void scalarSubtract(float scalar, float[] array, float[] output) {
        ScalarSubtract.compute(scalar, array, output);
    }
    
    /**
     * Element-wise square: output[i] = input[i] * input[i]
     * Used in Adam optimizer for computing gradient squares.
     */
    public static void elementwiseSquare(float[] input, float[] output) {
        ElementwiseSquare.compute(input, output);
    }
    
    /**
     * Element-wise square root: output[i] = sqrt(input[i])
     * Used in Adam optimizer denominator computation.
     */
    public static void elementwiseSqrt(float[] input, float[] output) {
        ElementwiseSqrt.compute(input, output);
    }
    
    /**
     * Element-wise square root with epsilon: output[i] = sqrt(input[i] + epsilon)
     * Common pattern in Adam optimizer to avoid division by zero.
     */
    public static void elementwiseSqrtWithEpsilon(float[] input, float epsilon, float[] output) {
        ElementwiseSqrt.computeWithEpsilon(input, epsilon, output);
    }
    
    /**
     * Element-wise scaling: output[i] = scale * input[i]
     * Used in Adam optimizer for bias correction and learning rate scaling.
     */
    public static void elementwiseScale(float[] input, float scale, float[] output) {
        ElementwiseScale.compute(input, scale, output);
    }
    
    /**
     * In-place element-wise scaling: array[i] = scale * array[i]
     * More memory efficient for temporary computations.
     */
    public static void elementwiseScaleInPlace(float[] array, float scale) {
        ElementwiseScale.computeInPlace(array, scale);
    }
    
    /**
     * Apply weight decay for regularization: weights[i] = weights[i] * (1 - decay_rate)
     * Used in AdamW optimizer for decoupled weight decay.
     */
    public static void weightDecay(float[] weights, float decayRate) {
        WeightDecay.compute(weights, decayRate);
    }
    
    /**
     * Apply weight decay to 2D weight matrix.
     * Commonly used for layer weights stored as [input][neuron] arrays.
     */
    public static void weightDecay(float[][] weights, float decayRate) {
        WeightDecay.compute(weights, decayRate);
    }
    
    /**
     * Exponential moving average in-place: current[i] = decay * current[i] + (1 - decay) * newValues[i]
     * Core operation for Adam momentum and velocity updates.
     */
    public static void exponentialMovingAverageInPlace(float[] current, float[] newValues, float decay) {
        ExponentialMovingAverage.computeInPlace(current, newValues, decay);
    }
    
    /**
     * Exponential moving average to output array: output[i] = decay * current[i] + (1 - decay) * newValues[i]
     */
    public static void exponentialMovingAverage(float[] current, float[] newValues, float decay, float[] output) {
        ExponentialMovingAverage.compute(current, newValues, decay, output);
    }
    
    /**
     * Element-wise division: output[i] = numerator[i] / denominator[i]
     */
    public static void elementwiseDivide(float[] numerator, float[] denominator, float[] output) {
        ElementwiseDivide.compute(numerator, denominator, output);
    }
    
    /**
     * Fused multiply-divide-subtract: params[i] -= scale * (numerator[i] / denominator[i])
     * Optimized for Adam parameter updates.
     */
    public static void fusedMultiplyDivideSubtract(float[] params, float[] numerator, float[] denominator, float scale) {
        FusedMultiplyDivideSubtract.compute(params, numerator, denominator, scale);
    }
    
    /**
     * Fused multiply-divide-add: params[i] += scale * (numerator[i] / denominator[i])
     */
    public static void fusedMultiplyDivideAdd(float[] params, float[] numerator, float[] denominator, float scale) {
        FusedMultiplyDivideSubtract.computeAdd(params, numerator, denominator, scale);
    }
    
    /**
     * Fused exponential moving average with gradient squaring: state = β * state + (1 - β) * gradient²
     * Optimized for second moment tracking in optimizers.
     */
    public static void fusedEMASquared(float[] state, float[] gradients, float beta) {
        FusedEMASquared.compute(state, gradients, beta);
    }
    
    /**
     * Fused bias correction with scaling: output[i] = scale * (input[i] / correction)
     */
    public static void fusedBiasCorrectScale(float[] input, float scale, float correction, float[] output) {
        FusedBiasCorrectScale.compute(input, scale, correction, output);
    }
    
    /**
     * Fused bias correction with scaling in-place: array[i] = scale * (array[i] / correction)
     */
    public static void fusedBiasCorrectScaleInPlace(float[] array, float scale, float correction) {
        FusedBiasCorrectScale.computeInPlace(array, scale, correction);
    }
    
    /**
     * Fused AdamW update - combines all AdamW operations in a single pass.
     * This is the ultimate optimization for AdamW, reducing 7+ memory passes to just 1.
     */
    public static void fusedAdamWUpdate(float[] params, float[] gradients, float[] momentum, float[] velocity,
                                      float beta1, float beta2, float learningRate, float epsilon,
                                      float weightDecay, float momentumCorrection, float velocityCorrection,
                                      boolean applyWeightDecay) {
        FusedAdamWUpdate.compute(params, gradients, momentum, velocity, beta1, beta2, learningRate,
                               epsilon, weightDecay, momentumCorrection, velocityCorrection, applyWeightDecay);
    }
    
    // ========== MATRIX OPERATIONS ==========
    
    /**
     * Outer product: output[i][j] = a[i] * b[j]
     */
    public static void matrixOuterProduct(float[] a, float[] b, float[][] output) {
        OuterProduct.compute(a, b, output);
    }
    
    /**
     * Pre-activations with column-major weights: output[neuron] = sum(input[i] * weights[i][neuron]) + bias[neuron]
     */
    public static void matrixPreActivationsColumnMajor(float[] inputs, float[][] weights, float[] biases, float[] output) {
        ColumnMajorPreActivations.compute(inputs, weights, biases, output);
    }
    
    /**
     * Compute weight gradients in column-major format: gradients[input][neuron] = input * delta[neuron]
     */
    public static void matrixWeightGradientsColumnMajor(float[] inputs, float[] neuronDeltas, float[][] weightGradients) {
        WeightGradientsColumnMajor.compute(inputs, neuronDeltas, weightGradients);
    }
    
    /**
     * Matrix-vector multiplication for column-major matrices: result[i] = sum(matrix[i][j] * vector[j])
     * More efficient than multiple dot product calls.
     */
    public static void matrixVectorMultiplyColumnMajor(float[][] matrix, float[] vector, float[] result) {
        MatrixVectorMultiplyColumnMajor.compute(matrix, vector, result);
    }
    
    // ========== WEIGHT INITIALIZATION ==========
    
    /**
     * Initialize weights using Xavier/Glorot initialization.
     * Good for sigmoid and tanh activation functions.
     */
    public static void weightInitXavier(float[][] weights, int fanIn, int fanOut, FastRandom random) {
        WeightInitXavier.compute(weights, fanIn, fanOut, random);
    }
    
    /**
     * Initialize weights using He initialization.
     * Good for ReLU and variants.
     */
    public static void weightInitHe(float[][] weights, int fanIn, FastRandom random) {
        WeightInitHe.compute(weights, fanIn, random);
    }

    /**
     * Initialize weights using He initialization plus uniform noise.
     * Good for ReLU and variants.
     */
    public static void weightInitHePlusUniformNoise(float[][] weights, int fanIn, float noiseLevel, FastRandom random) {
        WeightInitHe.compute(weights, fanIn, noiseLevel, random);
    }
    
    /**
     * Initialize embeddings with uniform distribution.
     * Recommended for embedding tables instead of He/Xavier initialization.
     * 
     * @param embeddings the embedding table to initialize
     * @param min minimum value (inclusive)
     * @param max maximum value (exclusive)
     */
    public static void embeddingInitUniform(float[][] embeddings, float min, float max, FastRandom random) {
        for (float[] row : embeddings) {
            random.fillUniform(row, min, max);
        }
    }
    
    // ========== BIAS INITIALIZATION ==========
    
    /**
     * Initialize biases to a constant value (typically 0.0f or 0.01f).
     */
    public static void biasInit(float[] biases, float value) {
        BiasInit.compute(biases, value);
    }
    
    /**
     * Initialize 2D matrix with a constant value.
     * Vectorized for optimal performance across entire matrix.
     */
    public static void matrixInit(float[][] matrix, float value) {
        MatrixInit.compute(matrix, value);
    }
    
    // ========== PARAMETER UPDATES ==========
    
    /**
     * Update neural network parameters using gradient descent.
     * 
     * <p>Core operation for all optimizers: param[i] = param[i] - learningRate * gradient[i]
     * 
     * <p>Used by optimizers to apply computed gradients to weights and biases.
     * Advanced optimizers may pre-process gradients (momentum, adaptive rates) before calling this.
     */
    public static void parameterUpdate(float[] parameters, float[] gradients, float learningRate) {
        ParameterUpdate.compute(parameters, gradients, learningRate);
    }
    
    /**
     * Lock-free parameter updates for parallel training using "Hogwild!" approach.
     * Allows race conditions on weight updates for maximum parallelism - mathematically sound for SGD.
     */
    public static void parameterUpdateAtomic(float[] parameters, float[] gradients, float learningRate) {
        ParameterUpdateAtomic.compute(parameters, gradients, learningRate);
    }
    
    // ========== LOSS FUNCTIONS ==========
    
    /**
     * Compute Mean Squared Error loss: (1/n) * sum((prediction - target)^2)
     */
    public static float lossComputeMSE(float[] predictions, float[] targets) {
        return MeanSquaredErrorLoss.computeLoss(predictions, targets);
    }
    
    /**
     * Compute MSE loss derivatives: (2/n) * (prediction - target)
     */
    public static void lossDerivativesMSE(float[] predictions, float[] targets, float[] output) {
        MeanSquaredErrorLoss.computeDerivatives(predictions, targets, output);
    }
    
    /**
     * Compute Cross-Entropy loss: -sum(trueLabels * log(predictions))
     */
    public static float lossComputeCrossEntropy(float[] trueLabels, float[] predictions) {
        return CrossEntropyLoss.compute(trueLabels, predictions);
    }
    
    /**
     * Compute Cross-Entropy loss gradient: predictions - trueLabels
     */
    public static void lossGradientCrossEntropy(float[] trueLabels, float[] predictions, float[] output) {
        CrossEntropyLoss.gradient(trueLabels, predictions, output);
    }
    
    /**
     * Compute Huber loss - robust loss function combining MSE and MAE.
     * Less sensitive to outliers than MSE.
     * 
     * @param predictions predicted values
     * @param targets true values
     * @param delta threshold parameter (typically 1.0)
     * @return average Huber loss
     */
    public static float lossComputeHuber(float[] predictions, float[] targets, float delta) {
        return HuberLoss.computeLoss(predictions, targets, delta);
    }
    
    /**
     * Compute Huber loss derivatives.
     * 
     * @param predictions predicted values
     * @param targets true values
     * @param delta threshold parameter
     * @param output pre-allocated array for derivatives
     */
    public static void lossDerivativesHuber(float[] predictions, float[] targets, float delta, float[] output) {
        HuberLoss.computeDerivatives(predictions, targets, delta, output);
    }
    
    // ========== ACTIVATION FUNCTIONS ==========
    
    /**
     * Apply Leaky ReLU activation: output[i] = input[i] > 0 ? input[i] : alpha * input[i]
     * 
     * @param input input array
     * @param alpha negative slope (typically 0.01 to 0.3)
     * @param output output array
     */
    public static void activationLeakyRelu(float[] input, float alpha, float[] output) {
        LeakyReLU.activate(input, alpha, output);
    }
    
    /**
     * Compute Leaky ReLU derivative: output[i] = input[i] > 0 ? 1.0f : alpha
     * 
     * @param input input array
     * @param alpha negative slope
     * @param output output array for derivatives
     */
    public static void activationLeakyReluDerivative(float[] input, float alpha, float[] output) {
        LeakyReLU.derivative(input, alpha, output);
    }
    
    // ========== BATCH OPERATIONS ==========
    
    /**
     * Batch matrix multiplication for processing multiple samples simultaneously.
     * Computes: output[batch][neuron] = sum(inputs[batch][input] * weights[input][neuron]) + biases[neuron]
     * 
     * @param inputs batch inputs [batchSize][inputSize]
     * @param weights weight matrix [inputSize][neurons] (column-major)
     * @param biases bias vector [neurons]
     * @param outputs pre-allocated output [batchSize][neurons]
     */
    public static void batchMatrixMultiply(float[][] inputs, float[][] weights, float[] biases, float[][] outputs) {
        BatchMatrixMultiply.compute(inputs, weights, biases, outputs);
    }
    
    /**
     * Batch matrix multiplication with parallelization across samples.
     * 
     * @param inputs batch inputs [batchSize][inputSize]
     * @param weights weight matrix [inputSize][neurons] (column-major)
     * @param biases bias vector [neurons]
     * @param outputs pre-allocated output [batchSize][neurons]
     * @param executor executor service for parallelization
     */
    public static void batchMatrixMultiplyParallel(float[][] inputs, float[][] weights, float[] biases, 
                                                   float[][] outputs, java.util.concurrent.ExecutorService executor) {
        BatchMatrixMultiply.computeParallel(inputs, weights, biases, outputs, executor);
    }
    
    /**
     * Average gradients across a batch for backpropagation.
     * 
     * @param batchGradients gradients from each sample [batchSize][gradientSize]
     * @param output pre-allocated output array for averaged gradients [gradientSize]
     */
    public static void batchAverageGradients(float[][] batchGradients, float[] output) {
        BatchGradientAccumulation.averageGradients(batchGradients, output);
    }
    
    /**
     * Average weight gradients across a batch.
     * 
     * @param batchWeightGradients gradients from each sample [batchSize][inputSize][neurons]
     * @param output pre-allocated output array [inputSize][neurons]
     */
    public static void batchAverageWeightGradients(float[][][] batchWeightGradients, float[][] output) {
        BatchGradientAccumulation.averageWeightGradients(batchWeightGradients, output);
    }
    
    /**
     * Compute batch weight gradients using outer product accumulation.
     * Efficiently computes and averages weight gradients across a batch.
     * 
     * @param batchInputs inputs for each sample [batchSize][inputSize]
     * @param batchNeuronDeltas neuron deltas for each sample [batchSize][neurons]
     * @param output pre-allocated output [inputSize][neurons]
     */
    public static void batchComputeWeightGradients(float[][] batchInputs, float[][] batchNeuronDeltas, 
                                                   float[][] output) {
        BatchGradientAccumulation.computeBatchWeightGradients(batchInputs, batchNeuronDeltas, output);
    }
    
    /**
     * Scale gradients by a factor (e.g., 1/batchSize for averaging).
     * In-place operation for efficiency.
     * 
     * @param gradients gradient array to scale
     * @param scale scaling factor
     */
    public static void scaleGradients(float[] gradients, float scale) {
        BatchGradientAccumulation.scaleGradients(gradients, scale);
    }
    
    // ===============================
    // GRADIENT NORM AND CLIPPING
    // ===============================
    
    /**
     * Compute L2 norm of gradient arrays.
     * 
     * @param array gradient array
     * @return L2 norm
     */
    public static float gradientNorm(float[] array) {
        return GradientNorm.computeNorm(array);
    }
    
    /**
     * Compute L2 norm of gradient matrix.
     * 
     * @param matrix gradient matrix
     * @return L2 norm
     */
    public static float gradientNorm(float[][] matrix) {
        return GradientNorm.computeNorm(matrix);
    }
    
    /**
     * Compute L2 norm across multiple weight matrices and bias vectors.
     * 
     * @param weights array of weight matrices (can contain nulls)
     * @param biases array of bias vectors (can contain nulls)
     * @return total L2 norm
     */
    public static float gradientNorm(float[][][] weights, float[][] biases) {
        return GradientNorm.computeNorm(weights, biases);
    }
    
    /**
     * Clip gradients by L2 norm. If norm exceeds maxNorm, all gradients are scaled down.
     * 
     * @param weights weight gradient matrices (modified in place)
     * @param biases bias gradient vectors (modified in place)
     * @param maxNorm maximum allowed L2 norm
     * @return scale factor applied (1.0 if no clipping)
     */
    public static float clipGradientsByNorm(float[][][] weights, float[][] biases, float maxNorm) {
        return GradientClipping.clipByNorm(weights, biases, maxNorm);
    }
    
    /**
     * Clip gradient values to be within [-maxValue, maxValue].
     * 
     * @param array gradient array (modified in place)
     * @param maxValue maximum absolute value allowed
     */
    public static void clipGradientsByValue(float[] array, float maxValue) {
        GradientClipping.clipByValue(array, maxValue);
    }
    
    /**
     * Scale a matrix in place by a factor.
     * 
     * @param matrix matrix to scale
     * @param scale scaling factor
     */
    public static void scaleMatrixInPlace(float[][] matrix, float scale) {
        GradientClipping.scaleInPlace(matrix, scale);
    }
    
    private NetMath() {} // Prevent instantiation
}