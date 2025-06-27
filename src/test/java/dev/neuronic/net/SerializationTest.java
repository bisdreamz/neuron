package dev.neuronic.net;

import dev.neuronic.net.optimizers.*;
import dev.neuronic.net.activators.*;
import dev.neuronic.net.outputs.*;
import dev.neuronic.net.layers.*;
import dev.neuronic.net.losses.*;
import dev.neuronic.net.activators.Activator;
import dev.neuronic.net.activators.LeakyReluActivator;
import dev.neuronic.net.layers.DenseLayer;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.outputs.HuberRegressionOutput;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

public class SerializationTest {
    
    @TempDir
    Path tempDir;
    
    @Test
    public void testLeakyReluSerialization() throws IOException {
        // Create network with LeakyReLU layers
        NeuralNet originalNet = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(5.0f)
            .layer(Layers.hiddenDenseLeakyRelu(20))          // Default alpha
            .layer(Layers.hiddenDenseLeakyRelu(15, 0.2f))    // Custom alpha
            .layer(Layers.hiddenDenseLeakyRelu(10, 0.05f))   // Different alpha
            .output(Layers.outputSoftmaxCrossEntropy(5));
        
        // Train with some data to ensure weights change
        float[] input = new float[10];
        float[] target = new float[5];
        for (int i = 0; i < 10; i++)
            input[i] = (float)(Math.random() - 0.5);
        
        target[2] = 1.0f; // One-hot target
        
        // Train a few steps
        for (int i = 0; i < 10; i++)
            originalNet.train(input, target);
        
        // Get predictions before saving
        float[] originalPrediction = originalNet.predict(input);
        
        // Save the network
        Path savedPath = tempDir.resolve("leakyrelu_test.nn");
        originalNet.save(savedPath);
        
        // Load the network
        NeuralNet loadedNet = NeuralNet.load(savedPath);
        
        // Get predictions after loading
        float[] loadedPrediction = loadedNet.predict(input);
        
        // Compare predictions
        assertEquals(originalPrediction.length, loadedPrediction.length);
        for (int i = 0; i < originalPrediction.length; i++)
            assertEquals(originalPrediction[i], loadedPrediction[i], 1e-6f,
                "Prediction mismatch at index " + i);
        
        // Verify LeakyReLU alphas were preserved
        Layer[] layers = loadedNet.getLayers();
        
        // First hidden layer should have default alpha (0.01)
        DenseLayer layer1 = (DenseLayer) layers[0];
        LeakyReluActivator activator1 = (LeakyReluActivator) getActivator(layer1);
        assertEquals(0.01f, activator1.getAlpha(), 1e-6f, "First layer alpha mismatch");
        
        // Second hidden layer should have alpha = 0.2
        DenseLayer layer2 = (DenseLayer) layers[1];
        LeakyReluActivator activator2 = (LeakyReluActivator) getActivator(layer2);
        assertEquals(0.2f, activator2.getAlpha(), 1e-6f, "Second layer alpha mismatch");
        
        // Third hidden layer should have alpha = 0.05
        DenseLayer layer3 = (DenseLayer) layers[2];
        LeakyReluActivator activator3 = (LeakyReluActivator) getActivator(layer3);
        assertEquals(0.05f, activator3.getAlpha(), 1e-6f, "Third layer alpha mismatch");
    }
    
    @Test
    public void testHuberLossSerialization() throws IOException {
        // Create networks with different Huber loss configurations
        NeuralNet[] originalNets = new NeuralNet[3];
        float[] deltas = {1.0f, 0.5f, 2.5f};
        
        for (int i = 0; i < 3; i++) {
            originalNets[i] = NeuralNet.newBuilder()
                .input(5)
                .setDefaultOptimizer(new SgdOptimizer(0.01f))
                .withGlobalGradientClipping(0.0f)
                .layer(Layers.hiddenDenseRelu(10))
                .output(Layers.outputHuberRegression(1, null, deltas[i]));
        }
        
        // Train each network
        float[][] inputs = {
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
            {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f},
            {0.5f, 1.5f, 2.5f, 3.5f, 4.5f}
        };
        float[][] targets = {{10.0f}, {-10.0f}, {5.0f}};
        
        for (int n = 0; n < 3; n++) {
            for (int epoch = 0; epoch < 20; epoch++) {
                for (int i = 0; i < inputs.length; i++)
                    originalNets[n].train(inputs[i], targets[i]);
            }
        }
        
        // Save and load each network
        for (int n = 0; n < 3; n++) {
            Path savedPath = tempDir.resolve("huber_test_" + n + ".nn");
            originalNets[n].save(savedPath);
            
            NeuralNet loadedNet = NeuralNet.load(savedPath);
            
            // Compare predictions
            for (int i = 0; i < inputs.length; i++) {
                float[] originalPred = originalNets[n].predict(inputs[i]);
                float[] loadedPred = loadedNet.predict(inputs[i]);
                
                assertEquals(originalPred[0], loadedPred[0], 1e-6f,
                    "Network " + n + " prediction mismatch for input " + i);
            }
            
            // Verify Huber delta was preserved
            HuberRegressionOutput outputLayer = (HuberRegressionOutput) loadedNet.getOutputLayer();
            float actualDelta = getDelta(outputLayer);
            assertEquals(deltas[n], actualDelta, 1e-6f, 
                "Huber delta mismatch for network " + n);
        }
    }
    
    @Test
    public void testMixedNetworkSerialization() throws IOException {
        // Create a complex network with both LeakyReLU and Huber loss
        NeuralNet originalNet = NeuralNet.newBuilder()
            .input(20)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(1.0f)
            .layer(Layers.hiddenDenseRelu(50))               // Regular ReLU
            .layer(Layers.hiddenDenseLeakyRelu(40, 0.1f))   // LeakyReLU
            .layer(Layers.dropout(0.2f))                     // Dropout
            .layer(Layers.hiddenDenseLeakyRelu(30))         // Default LeakyReLU
            .layer(Layers.hiddenDenseTanh(20))              // Tanh
            .output(Layers.outputHuberRegression(3, null, 1.5f)); // Huber with delta=1.5
        
        // Generate random data
        float[][] inputs = new float[10][20];
        float[][] targets = new float[10][3];
        
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 20; j++)
                inputs[i][j] = (float)(Math.random() * 2 - 1);
            
            for (int j = 0; j < 3; j++)
                targets[i][j] = (float)(Math.random() * 10 - 5);
        }
        
        // Train
        for (int epoch = 0; epoch < 5; epoch++) {
            for (int i = 0; i < inputs.length; i++)
                originalNet.train(inputs[i], targets[i]);
        }
        
        // Get predictions before saving
        float[][] originalPredictions = new float[inputs.length][];
        for (int i = 0; i < inputs.length; i++)
            originalPredictions[i] = originalNet.predict(inputs[i]);
        
        // Save
        Path savedPath = tempDir.resolve("mixed_network.nn");
        originalNet.save(savedPath);
        
        // Load
        NeuralNet loadedNet = NeuralNet.load(savedPath);
        
        // Compare all predictions
        for (int i = 0; i < inputs.length; i++) {
            float[] loadedPred = loadedNet.predict(inputs[i]);
            
            assertEquals(originalPredictions[i].length, loadedPred.length);
            for (int j = 0; j < loadedPred.length; j++)
                assertEquals(originalPredictions[i][j], loadedPred[j], 0.5f,  // Higher tolerance due to dropout randomness
                    "Prediction mismatch at sample " + i + ", output " + j);
        }
        
        // Verify layer types and parameters
        Layer[] layers = loadedNet.getLayers();
        
        // Layer 0: ReLU (checked implicitly)
        // Layer 1: LeakyReLU with alpha=0.1
        DenseLayer layer1 = (DenseLayer) layers[1];
        LeakyReluActivator leaky1 = (LeakyReluActivator) getActivator(layer1);
        assertEquals(0.1f, leaky1.getAlpha(), 1e-6f);
        
        // Layer 2: Dropout (checked implicitly)
        // Layer 3: LeakyReLU with default alpha
        DenseLayer layer3 = (DenseLayer) layers[3];
        LeakyReluActivator leaky3 = (LeakyReluActivator) getActivator(layer3);
        assertEquals(0.01f, leaky3.getAlpha(), 1e-6f);
        
        // Layer 4: Tanh (checked implicitly)
        // Layer 5: Huber output with delta=1.5
        HuberRegressionOutput outputLayer = (HuberRegressionOutput) layers[5];
        assertEquals(1.5f, getDelta(outputLayer), 1e-6f);
    }
    
    // Helper methods to access private fields for testing
    private Activator getActivator(DenseLayer layer) {
        try {
            var field = DenseLayer.class.getDeclaredField("activator");
            field.setAccessible(true);
            return (Activator) field.get(layer);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    private float getDelta(HuberRegressionOutput layer) {
        try {
            var field = HuberRegressionOutput.class.getDeclaredField("delta");
            field.setAccessible(true);
            return field.getFloat(layer);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}