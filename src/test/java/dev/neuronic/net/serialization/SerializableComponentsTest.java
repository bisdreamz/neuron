package dev.neuronic.net.serialization;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.activators.ReluActivator;
import dev.neuronic.net.layers.DenseLayer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.outputs.SoftmaxCrossEntropyOutput;
import dev.neuronic.net.WeightInitStrategy;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test for all classes implementing Serializable interface.
 * Ensures that every serializable component can round-trip correctly.
 */
class SerializableComponentsTest {
    
    private static final int TEST_VERSION = SerializationConstants.CURRENT_VERSION;
    
    @Test
    void testSgdOptimizerSerialization() throws IOException {
        // Test SGD optimizer serialization
        SgdOptimizer original = new SgdOptimizer(0.003f);
        
        // Serialize
        byte[] serialized = serializeComponent(original);
        
        // Deserialize
        SgdOptimizer deserialized = SgdOptimizer.deserialize(
            new DataInputStream(new ByteArrayInputStream(serialized)), TEST_VERSION);
        
        // Verify
        assertEquals(original.getLearningRate(), deserialized.getLearningRate(), 1e-7f,
            "Learning rate should be preserved");
        
        // Test functional equivalence with dummy data
        float[][] weights = {{0.1f, 0.2f}, {0.3f, 0.4f}};
        float[] biases = {0.1f, 0.2f};
        float[][] gradients = {{0.01f, 0.02f}, {0.03f, 0.04f}};
        float[] biasGradients = {0.01f, 0.02f};
        
        // Copy arrays for comparison
        float[][] originalWeights = deepCopy(weights);
        float[] originalBiases = Arrays.copyOf(biases, biases.length);
        float[][] deserializedWeights = deepCopy(weights);
        float[] deserializedBiases = Arrays.copyOf(biases, biases.length);
        
        // Apply updates
        original.optimize(originalWeights, originalBiases, gradients, biasGradients);
        deserialized.optimize(deserializedWeights, deserializedBiases, gradients, biasGradients);
        
        // Results should be identical
        assertArrayEquals(flatten(originalWeights), flatten(deserializedWeights), 1e-7f,
            "Weight updates should be identical");
        assertArrayEquals(originalBiases, deserializedBiases, 1e-7f,
            "Bias updates should be identical");
    }
    
    @Test
    void testDenseLayerSerialization() throws IOException {
        // Create a dense layer with specific weights
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        DenseLayer original = new DenseLayer(optimizer, ReluActivator.INSTANCE, 3, 2, WeightInitStrategy.XAVIER);
        
        // Train it a bit to get non-zero weights
        float[] input = {1.0f, 2.0f};
        float[] target = {0.1f, 0.8f, 0.1f};
        for (int i = 0; i < 5; i++) {
            var context = original.forward(input);
            original.backward(new DenseLayer.LayerContext[]{context}, 0, target);
        }
        
        // Get original prediction
        float[] originalOutput = original.forward(input).outputs();
        
        // Serialize
        byte[] serialized = serializeComponent(original);
        
        // Deserialize
        DenseLayer deserialized = DenseLayer.deserialize(
            new DataInputStream(new ByteArrayInputStream(serialized)), TEST_VERSION);
        
        // Test prediction equivalence
        float[] deserializedOutput = deserialized.forward(input).outputs();
        
        assertArrayEquals(originalOutput, deserializedOutput, 1e-6f,
            "Dense layer outputs should be identical after serialization");
        
        // Verify layer properties
        assertEquals(original.getOutputSize(), deserialized.getOutputSize(),
            "Output sizes should match");
    }
    
    @Test
    void testSoftmaxCrossEntropyOutputSerialization() throws IOException {
        // Create output layer
        SgdOptimizer optimizer = new SgdOptimizer(0.02f);
        SoftmaxCrossEntropyOutput original = new SoftmaxCrossEntropyOutput(
            optimizer, 3, 4, WeightInitStrategy.XAVIER);
        
        // Train it a bit
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] target = {0.0f, 1.0f, 0.0f}; // One-hot for class 1
        for (int i = 0; i < 10; i++) {
            var context = original.forward(input);
            original.backward(new SoftmaxCrossEntropyOutput.LayerContext[]{context}, 0, target);
        }
        
        // Get original predictions and loss
        var originalContext = original.forward(input);
        float[] originalOutput = originalContext.outputs();
        float originalLoss = original.computeLoss(originalOutput, target);
        
        // Serialize
        byte[] serialized = serializeComponent(original);
        
        // Deserialize
        SoftmaxCrossEntropyOutput deserialized = SoftmaxCrossEntropyOutput.deserialize(
            new DataInputStream(new ByteArrayInputStream(serialized)), TEST_VERSION);
        
        // Test prediction and loss equivalence
        var deserializedContext = deserialized.forward(input);
        float[] deserializedOutput = deserializedContext.outputs();
        float deserializedLoss = deserialized.computeLoss(deserializedOutput, target);
        
        assertArrayEquals(originalOutput, deserializedOutput, 1e-6f,
            "Softmax outputs should be identical after serialization");
        assertEquals(originalLoss, deserializedLoss, 1e-6f,
            "Loss computation should be identical after serialization");
        
        // Verify layer properties
        assertEquals(original.getOutputSize(), deserialized.getOutputSize(),
            "Output sizes should match");
    }
    
    @Test
    void testNeuralNetSerialization() throws IOException {
        // Create a multi-layer network
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        NeuralNet original = NeuralNet.newBuilder()
            .input(4)
            .layer(DenseLayer.spec(6, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE))
            .layer(DenseLayer.spec(4, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE))
            .output(SoftmaxCrossEntropyOutput.spec(3, optimizer, WeightInitStrategy.XAVIER));
        
        // Train it for several iterations
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] target = {0.0f, 1.0f, 0.0f};
        for (int i = 0; i < 20; i++) {
            original.train(input, target);
        }
        
        // Get original prediction
        float[] originalOutput = original.predict(input);
        
        // Serialize
        byte[] serialized = serializeComponent(original);
        
        // Deserialize
        NeuralNet deserialized = NeuralNet.deserialize(
            new DataInputStream(new ByteArrayInputStream(serialized)), TEST_VERSION);
        
        // Test prediction equivalence
        float[] deserializedOutput = deserialized.predict(input);
        
        assertArrayEquals(originalOutput, deserializedOutput, 1e-6f,
            "Neural network outputs should be identical after serialization");
    }
    
    @Test
    void testSerializationSizeConsistency() throws IOException {
        // Test that getSerializedSize() matches actual serialization size
        
        // Test SGD optimizer
        SgdOptimizer sgd = new SgdOptimizer(0.01f);
        byte[] sgdSerialized = serializeComponent(sgd);
        assertEquals(sgd.getSerializedSize(TEST_VERSION), sgdSerialized.length,
            "SGD serialized size should match getSerializedSize()");
        
        // Test Dense layer
        DenseLayer dense = new DenseLayer(sgd, ReluActivator.INSTANCE, 3, 2, WeightInitStrategy.XAVIER);
        byte[] denseSerialized = serializeComponent(dense);
        assertEquals(dense.getSerializedSize(TEST_VERSION), denseSerialized.length,
            "Dense layer serialized size should match getSerializedSize()");
        
        // Test Output layer
        SoftmaxCrossEntropyOutput output = new SoftmaxCrossEntropyOutput(sgd, 3, 4, WeightInitStrategy.XAVIER);
        byte[] outputSerialized = serializeComponent(output);
        assertEquals(output.getSerializedSize(TEST_VERSION), outputSerialized.length,
            "Output layer serialized size should match getSerializedSize()");
    }
    
    @Test
    void testTypeIdConsistency() {
        // Verify all components have unique type IDs
        SgdOptimizer sgd = new SgdOptimizer(0.01f);
        DenseLayer dense = new DenseLayer(sgd, ReluActivator.INSTANCE, 3, 2, WeightInitStrategy.XAVIER);
        SoftmaxCrossEntropyOutput output = new SoftmaxCrossEntropyOutput(sgd, 3, 4, WeightInitStrategy.XAVIER);
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .output(SoftmaxCrossEntropyOutput.spec(3, sgd, WeightInitStrategy.XAVIER));
        
        // All type IDs should be different
        int[] typeIds = {
            sgd.getTypeId(),
            dense.getTypeId(), 
            output.getTypeId(),
            net.getTypeId()
        };
        
        for (int i = 0; i < typeIds.length; i++) {
            for (int j = i + 1; j < typeIds.length; j++) {
                assertNotEquals(typeIds[i], typeIds[j], 
                    "Type IDs should be unique: " + typeIds[i] + " vs " + typeIds[j]);
            }
        }
        
        // All type IDs should be positive
        for (int typeId : typeIds) {
            assertTrue(typeId >= 0, "Type ID should be non-negative: " + typeId);
        }
    }
    
    @Test
    void testVersionCompatibility() throws IOException {
        // Test that serialization works with the current version
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        // Serialize with current version
        byte[] serialized = serializeComponent(optimizer);
        
        // Deserialize with same version
        SgdOptimizer deserialized = SgdOptimizer.deserialize(
            new DataInputStream(new ByteArrayInputStream(serialized)), TEST_VERSION);
        
        assertEquals(optimizer.getLearningRate(), deserialized.getLearningRate(), 1e-7f);
        
        // Version should be valid
        assertTrue(TEST_VERSION > 0, "Version should be positive");
        assertEquals(SerializationConstants.CURRENT_VERSION, TEST_VERSION,
            "Test version should match current version");
    }
    
    // Helper methods
    
    private byte[] serializeComponent(Serializable component) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(baos);
        component.writeTo(out, TEST_VERSION);
        out.close();
        return baos.toByteArray();
    }
    
    private float[][] deepCopy(float[][] array) {
        float[][] copy = new float[array.length][];
        for (int i = 0; i < array.length; i++) {
            copy[i] = Arrays.copyOf(array[i], array[i].length);
        }
        return copy;
    }
    
    private float[] flatten(float[][] array) {
        int totalLength = Arrays.stream(array).mapToInt(arr -> arr.length).sum();
        float[] flattened = new float[totalLength];
        int index = 0;
        for (float[] row : array) {
            System.arraycopy(row, 0, flattened, index, row.length);
            index += row.length;
        }
        return flattened;
    }
}