package dev.neuronic.net.serialization;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.activators.Activator;
import dev.neuronic.net.activators.ReluActivator;
import dev.neuronic.net.layers.DenseLayer;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.outputs.SoftmaxCrossEntropyOutput;
import dev.neuronic.net.WeightInitStrategy;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.ExecutorService;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test custom type registration and serialization.
 */
class CustomTypeSerializationTest {
    
    @TempDir
    Path tempDir;
    
    /**
     * Custom activator for testing registration.
     */
    static class CustomActivator implements Activator, Serializable {
        public static final CustomActivator INSTANCE = new CustomActivator();
        
        @Override
        public void activate(float[] input, float[] output) {
            // Simple custom activation: x^2 (for testing purposes)
            for (int i = 0; i < input.length; i++) {
                output[i] = input[i] * input[i];
            }
        }
        
        @Override
        public void derivative(float[] preActivations, float[] derivatives) {
            // Derivative: 2x
            for (int i = 0; i < preActivations.length; i++) {
                derivatives[i] = 2 * preActivations[i];
            }
        }
        
        @Override
        public void activate(float[] input, float[] output, ExecutorService executor) {
            activate(input, output); // Simple fallback
        }
        
        @Override
        public void derivative(float[] preActivations, float[] derivatives, ExecutorService executor) {
            derivative(preActivations, derivatives); // Simple fallback
        }
        
        @Override
        public void writeTo(DataOutputStream out, int version) throws IOException {
            // No state to serialize
        }
        
        @Override
        public void readFrom(DataInputStream in, int version) throws IOException {
            // No state to deserialize
        }
        
        @Override
        public int getSerializedSize(int version) {
            return 0; // No state
        }
        
        @Override
        public int getTypeId() {
            return SerializationConstants.TYPE_CUSTOM; // Will use class name
        }
    }
    
    /**
     * Custom optimizer for testing registration.
     */
    static class CustomOptimizer implements Optimizer, Serializable {
        private final float factor;
        
        public CustomOptimizer(float factor) {
            this.factor = factor;
        }
        
        @Override
        public void optimize(float[][] weights, float[] biases, float[][] weightGradients, float[] biasGradients) {
            // Simple custom update: subtract factor * gradient
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] -= factor * weightGradients[i][j];
                }
            }
            for (int i = 0; i < biases.length; i++) {
                biases[i] -= factor * biasGradients[i];
            }
        }
        
        @Override
        public void optimize(float[][] weights, float[] biases, float[][] weightGradients, float[] biasGradients, ExecutorService executor) {
            optimize(weights, biases, weightGradients, biasGradients); // Simple fallback
        }
        
        @Override
        public void setLearningRate(float learningRate) {
            // This is a test optimizer, learning rate is not used
        }
        
        @Override
        public void writeTo(DataOutputStream out, int version) throws IOException {
            out.writeFloat(factor);
        }
        
        @Override
        public void readFrom(DataInputStream in, int version) throws IOException {
            throw new UnsupportedOperationException("Use factory method");
        }
        
        public static CustomOptimizer deserialize(DataInputStream in, int version) throws IOException {
            float factor = in.readFloat();
            return new CustomOptimizer(factor);
        }
        
        @Override
        public int getSerializedSize(int version) {
            return 4; // float factor
        }
        
        @Override
        public int getTypeId() {
            return SerializationConstants.TYPE_CUSTOM; // Will use class name
        }
        
        public float getFactor() {
            return factor;
        }
    }
    
    @Test
    void testCustomActivatorRegistration() {
        // Register custom activator
        SerializationRegistry.registerActivator("CustomActivator", () -> CustomActivator.INSTANCE);
        
        // Verify registration
        assertTrue(SerializationRegistry.isActivatorRegistered("CustomActivator"));
        
        // Test creation
        Activator created = SerializationRegistry.createActivator("CustomActivator");
        assertEquals(CustomActivator.INSTANCE, created);
        
        // Test that it's found as registered
        String registeredName = SerializationRegistry.getRegisteredName(CustomActivator.INSTANCE);
        assertEquals("CustomActivator", registeredName);
    }
    
    @Test 
    void testCustomOptimizerRegistration() {
        // Register custom optimizer
        SerializationRegistry.registerOptimizer("CustomOptimizer", 
            (in, version) -> CustomOptimizer.deserialize(in, version));
        
        // Verify registration
        assertTrue(SerializationRegistry.isOptimizerRegistered("CustomOptimizer"));
        
        // Create original optimizer
        CustomOptimizer original = new CustomOptimizer(0.05f);
        
        // Register the instance for name lookup
        SerializationRegistry.registerOptimizer("CustomOptimizer", 
            (in, version) -> CustomOptimizer.deserialize(in, version));
        
        // Note: For this test to work fully, we'd need to register the instance in the reverse lookup map
        // For now, we just test the basic registration functionality
    }
    
    @Test
    void testSerializationWithBuiltInTypesOnly() throws IOException {
        // Test that built-in types work without any custom registration
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .layer(DenseLayer.spec(4, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE))
            .output(SoftmaxCrossEntropyOutput.spec(2, optimizer, WeightInitStrategy.XAVIER));
        
        // Test input
        float[] input = {1.0f, 2.0f, 3.0f};
        float[] originalOutput = net.predict(input);
        
        // Serialize and deserialize
        Path modelFile = tempDir.resolve("builtin_types.bin");
        ModelSerializer.save(net, modelFile);
        NeuralNet loadedNet = ModelSerializer.load(modelFile);
        
        // Verify identical behavior
        float[] loadedOutput = loadedNet.predict(input);
        assertArrayEquals(originalOutput, loadedOutput, 1e-6f,
            "Built-in types should serialize/deserialize correctly");
    }
    
    @Test
    void testRegistryIsolation() {
        // Test that registry operations don't interfere with each other
        
        // Different types should be independent
        assertFalse(SerializationRegistry.isLayerRegistered("NonExistentLayer"));
        assertFalse(SerializationRegistry.isActivatorRegistered("NonExistentActivator"));
        assertFalse(SerializationRegistry.isOptimizerRegistered("NonExistentOptimizer"));
        
        // Register one type
        SerializationRegistry.registerActivator("TestActivator", () -> ReluActivator.INSTANCE);
        
        // Other types should still be unregistered
        assertFalse(SerializationRegistry.isLayerRegistered("TestActivator"));
        assertFalse(SerializationRegistry.isOptimizerRegistered("TestActivator"));
        assertTrue(SerializationRegistry.isActivatorRegistered("TestActivator"));
    }
    
    @Test
    void testCustomTypeErrorHandling() {
        // Test error cases for custom types
        
        // Unknown activator
        assertThrows(IllegalArgumentException.class, () -> {
            SerializationRegistry.createActivator("UnknownActivator");
        });
        
        // Unknown layer  
        assertThrows(IllegalArgumentException.class, () -> {
            SerializationRegistry.createLayer("UnknownLayer", null, 1);
        });
        
        // Unknown optimizer
        assertThrows(IllegalArgumentException.class, () -> {
            SerializationRegistry.createOptimizer("UnknownOptimizer", null, 1);
        });
    }
}