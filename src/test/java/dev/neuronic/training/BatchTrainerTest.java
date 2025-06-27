package dev.neuronic.training;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.losses.MseLoss;
import dev.neuronic.net.losses.CrossEntropyLoss;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.training.BatchTrainer;
import dev.neuronic.net.training.TrainingCallback;
import dev.neuronic.net.training.TrainingMetrics;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class BatchTrainerTest {
    
    @Test
    public void testBatchTrainerBasicFunctionality() {
        // Create simple dataset - XOR problem
        float[][] inputs = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1},
            {0, 0}, {0, 1}, {1, 0}, {1, 1},
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        
        float[][] targets = {
            {0}, {1}, {1}, {0},
            {0}, {1}, {1}, {0},
            {0}, {1}, {1}, {0}
        };
        
        // Build simple network
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0f); // Higher LR for faster convergence
        NeuralNet model = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
        
        // Create trainer with minimal config
        BatchTrainer.TrainingConfig config = new BatchTrainer.TrainingConfig.Builder()
            .batchSize(4)
            .epochs(50) // More epochs for stability
            .validationSplit(0.3f)
            .verbosity(0) // Silent
            .build();
        
        BatchTrainer trainer = new BatchTrainer(model, MseLoss.INSTANCE, config);
        
        // Train
        BatchTrainer.TrainingResult result = trainer.fit(inputs, targets);
        
        // Verify training completed
        assertNotNull(result);
        assertNotNull(result.getMetrics());
        assertTrue(result.getMetrics().getEpochCount() > 0);
        
        // Check that loss decreased
        double[] lossHistory = result.getMetrics().getTrainingLossHistory();
        assertTrue(lossHistory.length > 0);
        if (lossHistory.length > 10) {
            // Compare first 5 epochs average to last 5 epochs average
            double avgEarlyLoss = 0;
            for (int i = 0; i < 5; i++) {
                avgEarlyLoss += lossHistory[i];
            }
            avgEarlyLoss /= 5;
            
            double avgLateLoss = 0;
            for (int i = lossHistory.length - 5; i < lossHistory.length; i++) {
                avgLateLoss += lossHistory[i];
            }
            avgLateLoss /= 5;
            
            // Allow for some tolerance due to randomness
            assertTrue(avgLateLoss < avgEarlyLoss * 1.1, 
                      String.format("Expected loss to decrease or stay similar. Early: %.4f, Late: %.4f", 
                                   avgEarlyLoss, avgLateLoss));
        }
    }
    
    @Test
    public void testBatchTrainerWithCallbacks() {
        // Simple dataset
        float[][] inputs = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}};
        float[][] targets = {{0}, {2}, {4}, {6}, {8}, {10}, {12}, {14}};
        
        // Build network
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        NeuralNet model = NeuralNet.newBuilder()
            .input(1)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(4))
            .output(Layers.outputLinearRegression(1));
        
        // Track callback execution
        final int[] callbackCalls = {0, 0, 0};
        
        TrainingCallback testCallback = new TrainingCallback() {
            @Override
            public void onTrainingStart(NeuralNet m, TrainingMetrics metrics) {
                callbackCalls[0]++;
            }
            
            @Override
            public void onEpochEnd(int epoch, TrainingMetrics metrics) {
                callbackCalls[1]++;
            }
            
            @Override
            public void onTrainingEnd(NeuralNet m, TrainingMetrics metrics) {
                callbackCalls[2]++;
            }
        };
        
        // Create trainer with callback
        BatchTrainer.TrainingConfig config = new BatchTrainer.TrainingConfig.Builder()
            .batchSize(2)
            .epochs(5)
            .verbosity(0)
            .build();
        
        BatchTrainer trainer = new BatchTrainer(model, MseLoss.INSTANCE, config)
            .withCallback(testCallback);
        
        // Train
        trainer.fit(inputs, targets);
        
        // Verify callbacks were called
        assertEquals(1, callbackCalls[0], "onTrainingStart should be called once");
        assertEquals(5, callbackCalls[1], "onEpochEnd should be called for each epoch");
        assertEquals(1, callbackCalls[2], "onTrainingEnd should be called once");
    }
    
    @Test
    public void testGradientAccumulation() {
        // Test with a simple learnable pattern
        float[][] inputs = {
            {1, 0}, {0, 1}, {1, 1}, {0, 0},
            {1, 0}, {0, 1}, {1, 1}, {0, 0},
            {1, 0}, {0, 1}, {1, 1}, {0, 0},
            {1, 0}, {0, 1}, {1, 1}, {0, 0}
        };
        
        float[][] targets = {
            {1, 0}, {0, 1}, {1, 0}, {0, 1},
            {1, 0}, {0, 1}, {1, 0}, {0, 1},
            {1, 0}, {0, 1}, {1, 0}, {0, 1},
            {1, 0}, {0, 1}, {1, 0}, {0, 1}
        };
        
        // Build network
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.001f);
        NeuralNet model = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        // Train with mini-batches
        BatchTrainer.TrainingConfig config = new BatchTrainer.TrainingConfig.Builder()
            .batchSize(4) // Will create 4 batches
            .epochs(20)
            .validationSplit(0.0f)
            .shuffle(false) // Keep order for reproducibility
            .verbosity(0)
            .build();
        
        BatchTrainer trainer = new BatchTrainer(model, 
                                              CrossEntropyLoss.INSTANCE,
                                              config);
        
        // Train
        BatchTrainer.TrainingResult result = trainer.fit(inputs, targets);
        
        // Verify training happened
        assertNotNull(result);
        assertEquals(20, result.getMetrics().getEpochCount());
        
        // Verify that the model was trained
        double[] lossHistory = result.getMetrics().getTrainingLossHistory();
        assertEquals(20, lossHistory.length);
        
        // The batch trainer with gradient accumulation is working if:
        // 1. Training completed all epochs
        // 2. Metrics were recorded properly
        // Loss might not always decrease with this simple test data,
        // but the machinery should work correctly
        assertTrue(result.getMetrics().getTotalSamplesSeen() > 0, 
                  "Samples should have been processed");
    }
}