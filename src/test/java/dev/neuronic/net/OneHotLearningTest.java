package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class OneHotLearningTest {
    
    @Test
    public void testOneHotOnlyLearning() {
        // Test with ONLY one-hot features to isolate from embedding issues
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0f);
        
        Feature[] features = {
            Feature.oneHot(4, "device"),
            Feature.oneHot(3, "connection"),
            Feature.passthrough("bid_floor")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(16, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Create training data - different combinations should lead to different outputs
        Map<String, Object>[] inputs = new Map[4];
        float[] targets = new float[4];
        
        // Device 0, Connection 0 -> low value
        inputs[0] = new HashMap<>();
        inputs[0].put("device", 0);
        inputs[0].put("connection", 0);
        inputs[0].put("bid_floor", 0.1f);
        targets[0] = 1.0f;
        
        // Device 1, Connection 1 -> medium value
        inputs[1] = new HashMap<>();
        inputs[1].put("device", 1);
        inputs[1].put("connection", 1);
        inputs[1].put("bid_floor", 0.2f);
        targets[1] = 2.0f;
        
        // Device 2, Connection 2 -> high value
        inputs[2] = new HashMap<>();
        inputs[2].put("device", 2);
        inputs[2].put("connection", 2);
        inputs[2].put("bid_floor", 0.3f);
        targets[2] = 3.0f;
        
        // Device 3, Connection 0 -> medium-high value
        inputs[3] = new HashMap<>();
        inputs[3].put("device", 3);
        inputs[3].put("connection", 0);
        inputs[3].put("bid_floor", 0.4f);
        targets[3] = 2.5f;
        
        // Get initial predictions
        float[] initialPreds = new float[4];
        for (int i = 0; i < 4; i++) {
            initialPreds[i] = model.predictFloat(inputs[i]);
            System.out.println("Initial pred[" + i + "] = " + initialPreds[i]);
        }
        
        // Train for multiple epochs
        for (int epoch = 0; epoch < 100; epoch++) {
            for (int i = 0; i < 4; i++) {
                model.train(inputs[i], targets[i]);
            }
            
            if (epoch % 20 == 0) {
                System.out.println("Epoch " + epoch);
                for (int i = 0; i < 4; i++) {
                    float pred = model.predictFloat(inputs[i]);
                    System.out.printf("  Sample %d: target=%.1f, pred=%.3f%n", i, targets[i], pred);
                }
            }
        }
        
        // Get final predictions
        float[] finalPreds = new float[4];
        for (int i = 0; i < 4; i++) {
            finalPreds[i] = model.predictFloat(inputs[i]);
        }
        
        // Check that model learned something
        for (int i = 0; i < 4; i++) {
            float initialError = Math.abs(initialPreds[i] - targets[i]);
            float finalError = Math.abs(finalPreds[i] - targets[i]);
            System.out.printf("Sample %d: Initial error=%.3f, Final error=%.3f%n", 
                            i, initialError, finalError);
            assertTrue(finalError < initialError, 
                      "Model should improve prediction for sample " + i);
        }
        
        // Check predictions are diverse (not all the same)
        float minPred = Float.MAX_VALUE, maxPred = Float.MIN_VALUE;
        for (float pred : finalPreds) {
            minPred = Math.min(minPred, pred);
            maxPred = Math.max(maxPred, pred);
        }
        assertTrue(maxPred - minPred > 0.5f, 
                  "Predictions should be diverse, range: " + (maxPred - minPred));
    }
    
    @Test 
    public void testMixedWithoutHashedEmbeddings() {
        // Test with regular embeddings + one-hot (no hashed embeddings)
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0f);
        
        Feature[] features = {
            Feature.embedding(100, 8, "item_id"),  // Regular embedding
            Feature.oneHot(4, "device"),
            Feature.passthrough("bid_floor")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(16, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Test data
        Map<String, Object> input1 = new HashMap<>();
        input1.put("item_id", 10);
        input1.put("device", 1);
        input1.put("bid_floor", 0.5f);
        
        Map<String, Object> input2 = new HashMap<>();
        input2.put("item_id", 20);
        input2.put("device", 2);
        input2.put("bid_floor", 0.7f);
        
        float target1 = 2.0f;
        float target2 = 3.0f;
        
        // Initial predictions
        float init1 = model.predictFloat(input1);
        float init2 = model.predictFloat(input2);
        System.out.println("Initial predictions: " + init1 + ", " + init2);
        
        // Train
        for (int i = 0; i < 50; i++) {
            model.train(input1, target1);
            model.train(input2, target2);
        }
        
        // Final predictions
        float final1 = model.predictFloat(input1);
        float final2 = model.predictFloat(input2);
        System.out.println("Final predictions: " + final1 + ", " + final2);
        
        // Should learn different outputs
        assertTrue(Math.abs(final1 - target1) < Math.abs(init1 - target1), 
                  "Should improve prediction 1");
        assertTrue(Math.abs(final2 - target2) < Math.abs(init2 - target2), 
                  "Should improve prediction 2");
        assertTrue(Math.abs(final1 - final2) > 0.5f, 
                  "Should learn different predictions");
    }
}