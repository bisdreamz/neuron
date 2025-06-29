package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to reproduce the issue where one-hot only features in MixedFeatureInputLayer
 * result in all predictions being the same.
 */
public class OneHotOnlyMixedLayerTest {
    
    @Test
    public void testOneHotOnlyPredictionsConvergeToMean() {
        // Mimicking user's setup - MixedFeatureInputLayer with only one-hot features
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0001f);
        
        Feature[] features = {
            Feature.oneHot(5, "feature1"),
            Feature.oneHot(4, "feature2"),
            Feature.oneHot(3, "feature3")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(32, 0.01f))
            .layer(Layers.hiddenDenseLeakyRelu(16, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Generate random training data
        Random rand = new Random(42);
        Map<String, Object>[] inputs = new Map[100];
        float[] targets = new float[100];
        
        for (int i = 0; i < 100; i++) {
            inputs[i] = new HashMap<>();
            inputs[i].put("feature1", rand.nextInt(5));
            inputs[i].put("feature2", rand.nextInt(4));
            inputs[i].put("feature3", rand.nextInt(3));
            
            // Target is a function of the features
            int f1 = (int) inputs[i].get("feature1");
            int f2 = (int) inputs[i].get("feature2");
            int f3 = (int) inputs[i].get("feature3");
            targets[i] = f1 * 1.0f + f2 * 2.0f + f3 * 3.0f + 1.0f;
        }
        
        // Calculate mean target
        float meanTarget = 0;
        for (float t : targets) {
            meanTarget += t;
        }
        meanTarget /= targets.length;
        System.out.println("Mean target: " + meanTarget);
        
        // Initial predictions
        System.out.println("\nInitial predictions (sample):");
        for (int i = 0; i < 5; i++) {
            float pred = model.predictFloat(inputs[i]);
            System.out.printf("Sample %d: target=%.1f, pred=%.3f\n", i, targets[i], pred);
        }
        
        // Train for many epochs
        for (int epoch = 0; epoch < 100; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                model.train(inputs[i], targets[i]);
            }
            
            // Check predictions every 20 epochs
            if (epoch % 20 == 0) {
                System.out.println("\nEpoch " + epoch + " predictions:");
                float minPred = Float.MAX_VALUE;
                float maxPred = Float.MIN_VALUE;
                float avgPred = 0;
                
                for (int i = 0; i < 10; i++) {
                    float pred = model.predictFloat(inputs[i]);
                    minPred = Math.min(minPred, pred);
                    maxPred = Math.max(maxPred, pred);
                    avgPred += pred;
                    if (i < 5) {
                        System.out.printf("Sample %d: target=%.1f, pred=%.3f\n", i, targets[i], pred);
                    }
                }
                avgPred /= 10;
                System.out.printf("Prediction range: [%.3f, %.3f], avg=%.3f\n", minPred, maxPred, avgPred);
            }
        }
        
        // Final check - are all predictions similar?
        float[] finalPreds = new float[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            finalPreds[i] = model.predictFloat(inputs[i]);
        }
        
        // Calculate variance
        float mean = 0;
        for (float p : finalPreds) {
            mean += p;
        }
        mean /= finalPreds.length;
        
        float variance = 0;
        for (float p : finalPreds) {
            variance += (p - mean) * (p - mean);
        }
        variance /= finalPreds.length;
        float stdDev = (float) Math.sqrt(variance);
        
        System.out.println("\nFinal statistics:");
        System.out.println("Mean prediction: " + mean);
        System.out.println("Std deviation: " + stdDev);
        System.out.println("All converged to mean target? " + (Math.abs(mean - meanTarget) < 0.5f));
        
        // The model should learn different predictions, not all the same
        assertTrue(stdDev > 0.5f, "Predictions should have variation, not all be the same");
    }
    
    @Test
    public void testDebugOneHotGradients() {
        // Simple test to check if gradients flow properly
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f); // High LR, no weight decay
        
        Feature[] features = {
            Feature.oneHot(2, "binary")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Create inputs
        Map<String, Object> input0 = new HashMap<>();
        input0.put("binary", 0);
        
        Map<String, Object> input1 = new HashMap<>();
        input1.put("binary", 1);
        
        // Track initial predictions to verify learning
        float pred0Initial = model.predictFloat(input0);
        float pred1Initial = model.predictFloat(input1);
        
        // Train with feature=0 -> target=0
        model.train(input0, 0.0f);
        
        // Train with feature=1 -> target=1
        model.train(input1, 1.0f);
        
        // Check that training changed predictions
        float pred0Final = model.predictFloat(input0);
        float pred1Final = model.predictFloat(input1);
        
        System.out.println("Prediction changes:");
        System.out.printf("Input 0: %.6f -> %.6f (change: %.6f)\n", 
                        pred0Initial, pred0Final, pred0Final - pred0Initial);
        System.out.printf("Input 1: %.6f -> %.6f (change: %.6f)\n", 
                        pred1Initial, pred1Final, pred1Final - pred1Initial);
        
        float totalChange = Math.abs(pred0Final - pred0Initial) + Math.abs(pred1Final - pred1Initial);
        assertTrue(totalChange > 0.01f, "Predictions should change during training");
        
        // Check predictions
        float pred0 = model.predictFloat(input0);
        float pred1 = model.predictFloat(input1);
        System.out.println("\nPredictions:");
        System.out.println("Input 0 -> " + pred0);
        System.out.println("Input 1 -> " + pred1);
        
        assertTrue(Math.abs(pred0 - pred1) > 0.1f, "Different inputs should give different predictions");
    }
}