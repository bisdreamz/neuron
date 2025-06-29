package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test specifically for 8 one-hot features as described by the user:
 * country, ad format, device type, os, pubid, domain, zoneid, devconnection
 */
public class EightOneHotTest {
    
    @Test
    public void testEightOneHotFeatures() {
        // User's exact configuration
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0001f);
        
        Feature[] features = {
            Feature.oneHot(200, "country"),      // ~200 countries
            Feature.oneHot(10, "ad_format"),     // banner, video, native, etc
            Feature.oneHot(5, "device_type"),    // mobile, desktop, tablet, tv, other
            Feature.oneHot(10, "os"),            // ios, android, windows, mac, linux, etc
            Feature.oneHot(1000, "pubid"),       // publisher IDs (might be high cardinality)
            Feature.oneHot(5000, "domain"),      // domains (high cardinality)
            Feature.oneHot(1000, "zoneid"),      // zone IDs
            Feature.oneHot(4, "devconnection")   // wifi, 4g, 3g, other
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(128, 0.01f))
            .layer(Layers.hiddenDenseLeakyRelu(64, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Generate synthetic data
        Random rand = new Random(42);
        int numSamples = 1000;
        Map<String, Object>[] inputs = new Map[numSamples];
        float[] targets = new float[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            inputs[i] = new HashMap<>();
            inputs[i].put("country", rand.nextInt(200));
            inputs[i].put("ad_format", rand.nextInt(10));
            inputs[i].put("device_type", rand.nextInt(5));
            inputs[i].put("os", rand.nextInt(10));
            inputs[i].put("pubid", rand.nextInt(1000));
            inputs[i].put("domain", rand.nextInt(5000));
            inputs[i].put("zoneid", rand.nextInt(1000));
            inputs[i].put("devconnection", rand.nextInt(4));
            
            // Create a target that depends on features
            int device = (int) inputs[i].get("device_type");
            int format = (int) inputs[i].get("ad_format");
            int conn = (int) inputs[i].get("devconnection");
            targets[i] = device * 0.2f + format * 0.1f + conn * 0.5f + rand.nextFloat() * 0.1f;
        }
        
        // Calculate mean target
        float meanTarget = 0;
        for (float t : targets) {
            meanTarget += t;
        }
        meanTarget /= targets.length;
        System.out.println("Mean target: " + meanTarget);
        
        // Initial predictions
        float[] initialPreds = new float[10];
        System.out.println("Initial predictions:");
        for (int i = 0; i < 10; i++) {
            initialPreds[i] = model.predictFloat(inputs[i]);
            System.out.printf("  Sample %d: target=%.3f, pred=%.3f\n", i, targets[i], initialPreds[i]);
        }
        
        // Train
        System.out.println("\nTraining...");
        for (int epoch = 0; epoch < 50; epoch++) {
            // Shuffle and train
            for (int i = 0; i < numSamples; i++) {
                int idx = rand.nextInt(numSamples);
                model.train(inputs[idx], targets[idx]);
            }
            
            if (epoch % 10 == 0) {
                float error = 0;
                for (int i = 0; i < 100; i++) {
                    float pred = model.predictFloat(inputs[i]);
                    error += Math.abs(pred - targets[i]);
                }
                System.out.printf("Epoch %d: avg error = %.4f\n", epoch, error / 100);
            }
        }
        
        // Final predictions
        System.out.println("\nFinal predictions:");
        float minPred = Float.MAX_VALUE, maxPred = Float.MIN_VALUE;
        float totalError = 0;
        for (int i = 0; i < 10; i++) {
            float pred = model.predictFloat(inputs[i]);
            System.out.printf("  Sample %d: target=%.3f, pred=%.3f, error=%.3f\n", 
                            i, targets[i], pred, Math.abs(pred - targets[i]));
            minPred = Math.min(minPred, pred);
            maxPred = Math.max(maxPred, pred);
            totalError += Math.abs(pred - targets[i]);
        }
        
        System.out.printf("\nPrediction range: [%.3f, %.3f]\n", minPred, maxPred);
        System.out.println("Average error: " + (totalError / 10));
        
        assertTrue(maxPred - minPred > 0.5f, 
                  "Predictions should be diverse, not all the same");
        assertTrue(totalError / 10 < 0.5f,
                  "Model should learn to predict reasonably well");
    }
    
    @Test
    public void testConcurrentTrainingWithOneHot() throws Exception {
        System.out.println("\n=== CONCURRENT TRAINING TEST ===");
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0001f);
        
        Feature[] features = {
            Feature.oneHot(100, "feature1"),
            Feature.oneHot(50, "feature2"),
            Feature.oneHot(10, "feature3")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(32, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Simple data
        int numSamples = 100;
        Map<String, Object>[] inputs = new Map[numSamples];
        float[] targets = new float[numSamples];
        
        Random rand = new Random(42);
        for (int i = 0; i < numSamples; i++) {
            inputs[i] = new HashMap<>();
            inputs[i].put("feature1", i % 100);
            inputs[i].put("feature2", i % 50);
            inputs[i].put("feature3", i % 10);
            targets[i] = (i % 10) * 0.1f + (i % 50) * 0.01f;
        }
        
        // Train from multiple threads
        int numThreads = 8;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch latch = new CountDownLatch(numThreads);
        AtomicInteger trainCount = new AtomicInteger(0);
        
        long startTime = System.currentTimeMillis();
        
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    Random threadRand = new Random(threadId);
                    for (int iter = 0; iter < 1000; iter++) {
                        int idx = threadRand.nextInt(numSamples);
                        model.train(inputs[idx], targets[idx]);
                        trainCount.incrementAndGet();
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        
        latch.await(30, TimeUnit.SECONDS);
        executor.shutdown();
        
        long duration = System.currentTimeMillis() - startTime;
        System.out.println("Trained " + trainCount.get() + " times in " + duration + "ms");
        System.out.println("Rate: " + (trainCount.get() * 1000 / duration) + " trains/sec");
        
        // Check predictions
        float minPred = Float.MAX_VALUE, maxPred = Float.MIN_VALUE;
        for (int i = 0; i < 10; i++) {
            float pred = model.predictFloat(inputs[i]);
            minPred = Math.min(minPred, pred);
            maxPred = Math.max(maxPred, pred);
        }
        
        System.out.printf("Prediction range after concurrent training: [%.3f, %.3f]\n", 
                        minPred, maxPred);
        
        assertTrue(maxPred - minPred > 0.1f, 
                  "Concurrent training should still produce diverse predictions");
    }
}