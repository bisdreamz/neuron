package dev.neuronic.net.serialization;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.activators.ReluActivator;
import dev.neuronic.net.layers.DenseLayer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.outputs.SoftmaxCrossEntropyOutput;
import dev.neuronic.net.WeightInitStrategy;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test neural network model serialization and deserialization.
 */
class ModelSerializationTest {
    
    @TempDir
    Path tempDir;
    
    @Test
    void testBasicModelSerialization() throws IOException {
        // Create a simple neural network
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        NeuralNet originalNet = NeuralNet.newBuilder()
            .input(5)
            .layer(DenseLayer.spec(10, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE))
            .output(SoftmaxCrossEntropyOutput.spec(3, optimizer, WeightInitStrategy.XAVIER));
        
        // Test prediction before serialization
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] originalOutput = originalNet.predict(input);
        
        // Serialize the model using the new direct API
        Path modelFile = tempDir.resolve("test_model.bin");
        originalNet.save(modelFile);
        
        assertTrue(modelFile.toFile().exists(), "Model file should be created");
        assertTrue(modelFile.toFile().length() > 0, "Model file should not be empty");
        
        // Deserialize the model using the new direct API
        NeuralNet loadedNet = NeuralNet.load(modelFile);
        
        // Test prediction after deserialization
        float[] loadedOutput = loadedNet.predict(input);
        
        // Outputs should be identical
        assertArrayEquals(originalOutput, loadedOutput, 1e-6f, 
            "Loaded model should produce identical outputs");
    }
    
    @Test
    void testFileSizeEstimation() throws IOException {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(5)
            .layer(DenseLayer.spec(10, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE))
            .output(SoftmaxCrossEntropyOutput.spec(3, optimizer, WeightInitStrategy.XAVIER));
        
        long estimatedSize = ModelSerializer.estimateFileSize(net);
        
        Path modelFile = tempDir.resolve("size_test.bin");
        net.save(modelFile);
        
        long actualSize = modelFile.toFile().length();
        
        // Estimation should be reasonable (within 50% due to compression)
        assertTrue(actualSize <= estimatedSize, 
            "Compressed file should be smaller than uncompressed estimate");
        assertTrue(actualSize >= estimatedSize * 0.1, 
            "Compressed file shouldn't be too much smaller than estimate");
    }
    
    @Test
    void testCustomTypeRegistration() {
        // Test registering a custom activator
        SerializationRegistry.registerActivator("CustomActivator", () -> ReluActivator.INSTANCE);
        
        assertTrue(SerializationRegistry.isActivatorRegistered("CustomActivator"));
        assertFalse(SerializationRegistry.isActivatorRegistered("NonExistentActivator"));
        
        // Test creating registered activator
        var activator = SerializationRegistry.createActivator("CustomActivator");
        assertEquals(ReluActivator.INSTANCE, activator);
    }
}