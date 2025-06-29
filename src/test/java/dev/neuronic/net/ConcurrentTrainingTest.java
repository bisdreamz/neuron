package dev.neuronic.net;

import dev.neuronic.net.Layers;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

public class ConcurrentTrainingTest {
    
    @Test
    public void testConcurrentTrainingConvergence() throws Exception {
        // Create a simple network similar to the user's setup
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0001f);
        
        Feature[] features = {
            Feature.hashedEmbedding(50000, 16, "domain"),
            Feature.oneHot(4, "device"),
            Feature.passthrough("bidfloor")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .withGlobalGradientClipping(1f)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(32))
            .layer(Layers.dropout(0.1f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Create test data
        int numSamples = 1000;
        List<Map<String, Object>> inputs = new ArrayList<>();
        List<Float> targets = new ArrayList<>();
        
        Random rand = new Random(42);
        for (int i = 0; i < numSamples; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("domain", rand.nextInt(100));  // Hash values
            input.put("device", rand.nextInt(4));
            input.put("bidfloor", rand.nextFloat() * 5.0f);
            
            // Simple target function: bidfloor * 0.5 + device * 0.25
            float target = ((Float)input.get("bidfloor")) * 0.5f + ((Integer)input.get("device")) * 0.25f;
            
            inputs.add(input);
            targets.add(target);
        }
        
        // Train concurrently from multiple threads
        int numThreads = 8;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch latch = new CountDownLatch(numThreads);
        AtomicInteger trainCount = new AtomicInteger(0);
        
        // Measure initial predictions
        Set<Float> initialPredictions = new HashSet<>();
        for (int i = 0; i < 20; i++) {
            float pred = model.predictFloat(inputs.get(i));
            initialPredictions.add(pred);
        }
        
        System.out.println("Initial unique predictions: " + initialPredictions.size());
        
        // Concurrent training
        for (int t = 0; t < numThreads; t++) {
            executor.submit(() -> {
                try {
                    Random threadRand = new Random();
                    for (int iter = 0; iter < 100; iter++) {
                        // Random batch
                        List<Map<String, Object>> batch = new ArrayList<>();
                        List<Float> batchTargets = new ArrayList<>();
                        
                        for (int b = 0; b < 10; b++) {
                            int idx = threadRand.nextInt(numSamples);
                            batch.add(inputs.get(idx));
                            batchTargets.add(targets.get(idx));
                        }
                        
                        model.trainBatchMaps(batch, batchTargets);
                        trainCount.incrementAndGet();
                        
                        // Also do single training
                        int idx = threadRand.nextInt(numSamples);
                        model.train(inputs.get(idx), targets.get(idx));
                        trainCount.incrementAndGet();
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        
        latch.await(30, TimeUnit.SECONDS);
        executor.shutdown();
        
        System.out.println("Total training operations: " + trainCount.get());
        
        // Check predictions after training
        Set<Float> finalPredictions = new HashSet<>();
        float totalError = 0;
        int uniqueCount = 0;
        
        for (int i = 0; i < 100; i++) {
            float pred = model.predictFloat(inputs.get(i));
            float target = targets.get(i);
            totalError += Math.abs(pred - target);
            
            // Round to avoid floating point precision issues
            float rounded = Math.round(pred * 1000) / 1000f;
            finalPredictions.add(rounded);
            
            if (i < 10) {
                System.out.printf("Input %d: pred=%.3f, target=%.3f%n", i, pred, target);
            }
        }
        
        System.out.println("Final unique predictions: " + finalPredictions.size());
        System.out.println("Average error: " + (totalError / 100));
        
        // Assertions
        assertTrue(finalPredictions.size() > 10, 
            "Model should produce diverse predictions, got only " + finalPredictions.size() + " unique values");
        assertTrue(totalError / 100 < 2.0f, 
            "Model should learn something, average error too high: " + (totalError / 100));
    }
    
    @Test 
    public void testWeightUpdateConsistency() throws Exception {
        // Test that weights actually change during training
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0f); // No weight decay
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseLeakyRelu(16))
            .output(Layers.outputLinearRegression(1));
            
        // Get initial weights from first layer
        dev.neuronic.net.layers.Layer firstLayer = net.getLayers()[0];
        float[] initialWeights = Arrays.copyOf(firstLayer.getWeights(), 10); // Sample first 10
        
        // Train with known data
        float[][] inputs = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
        float[][] targets = {{3.0f}, {7.0f}, {11.0f}};
        
        for (int i = 0; i < 10; i++) {
            net.trainBatch(inputs, targets);
        }
        
        float[] finalWeights = Arrays.copyOf(firstLayer.getWeights(), 10);
        
        // Check that weights changed
        boolean weightsChanged = false;
        for (int i = 0; i < 10; i++) {
            if (Math.abs(initialWeights[i] - finalWeights[i]) > 0.0001f) {
                weightsChanged = true;
                break;
            }
        }
        
        assertTrue(weightsChanged, "Weights should change after training");
        
        // Check predictions are different for different inputs
        float pred1 = net.predict(new float[]{1.0f, 1.0f})[0];
        float pred2 = net.predict(new float[]{5.0f, 5.0f})[0];
        
        assertNotEquals(pred1, pred2, 0.001f, 
            "Different inputs should produce different predictions");
    }
}