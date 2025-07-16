package dev.neuronic.net.training;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.layers.DenseLayer;
import dev.neuronic.net.layers.DropoutLayer;
import dev.neuronic.net.Layers;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetInt;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import dev.neuronic.net.simple.SimpleNetTrainingResult;
import dev.neuronic.net.activators.ReluActivator;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.WeightInitStrategy;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to verify early stopping actually stops training when it should.
 */
public class EarlyStoppingTest {
    
    @Test
    public void testEarlyStoppingActuallyStops() {
        // Create a simple neural network
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f); // Higher learning rate for faster convergence
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(DenseLayer.spec(32, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE))
            .layer(DropoutLayer.spec(0.2f))
            .output(Layers.outputSoftmaxCrossEntropy(3, optimizer));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Generate dummy data that will plateau quickly
        Random rng = new Random(42);
        List<float[]> inputs = new ArrayList<>();
        List<Integer> targets = new ArrayList<>();
        
        // Create 1000 samples
        for (int i = 0; i < 1000; i++) {
            float[] input = new float[10];
            for (int j = 0; j < 10; j++) {
                input[j] = rng.nextFloat();
            }
            inputs.add(input);
            
            // Simple pattern: if first feature > 0.5, class 2, else random between 0 and 1
            if (input[0] > 0.5f) {
                targets.add(2);
            } else {
                targets.add(rng.nextInt(2)); // 0 or 1
            }
        }
        
        // Configure training with early stopping after 3 epochs without improvement
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .epochs(50)  // Max 50 epochs but should stop much earlier
            .batchSize(32)
            .validationSplit(0.2f)
            .withEarlyStopping(3)  // Stop after 3 epochs without improvement
            .verbosity(1)  // Enable output to see what happens
            .build();
        
        // Train and track results
        System.out.println("\n=== Testing Early Stopping ===");
        SimpleNetTrainingResult result = 
            classifier.trainBulk(inputs, targets, config);
        
        // Verify training stopped early
        int epochsTrained = result.getEpochsTrained();
        System.out.println("\nEpochs trained: " + epochsTrained);
        
        // With this simple pattern and early stopping patience of 3,
        // training should stop well before 50 epochs
        assertTrue(epochsTrained < 50, 
            "Training should have stopped early but ran all 50 epochs");
        
        // Typically with this pattern, it should stop within 5-10 epochs
        assertTrue(epochsTrained <= 15, 
            "Training should have stopped within 15 epochs but ran " + epochsTrained);
        
        System.out.println("✓ Early stopping worked correctly - stopped at epoch " + epochsTrained);
    }
    
    @Test
    public void testEarlyStoppingWithNoImprovement() {
        // Create a network that will immediately plateau
        AdamWOptimizer optimizer = new AdamWOptimizer(0.00001f, 0.01f); // Very low LR
        NeuralNet net = NeuralNet.newBuilder()
            .input(5)
            .setDefaultOptimizer(optimizer)
            .layer(DenseLayer.spec(8, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE))
            .output(Layers.outputSoftmaxCrossEntropy(2, optimizer));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Generate random data with no real pattern
        Random rng = new Random(123);
        List<float[]> inputs = new ArrayList<>();
        List<Integer> targets = new ArrayList<>();
        
        for (int i = 0; i < 200; i++) {
            float[] input = new float[5];
            for (int j = 0; j < 5; j++) {
                input[j] = rng.nextFloat();
            }
            inputs.add(input);
            targets.add(rng.nextInt(2));
        }
        
        // Configure with early stopping patience of 2
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .epochs(20)
            .batchSize(16)
            .validationSplit(0.3f)
            .withEarlyStopping(2)  // Very low patience
            .verbosity(0)  // Quiet mode
            .build();
        
        SimpleNetTrainingResult result = 
            classifier.trainBulk(inputs, targets, config);
        
        int epochsTrained = result.getEpochsTrained();
        
        // With random data and very low learning rate, should stop very quickly
        assertTrue(epochsTrained <= 5, 
            "With patience=2 and no improvement, should stop within 5 epochs but ran " + epochsTrained);
        
        System.out.println("✓ Early stopping with no improvement stopped at epoch " + epochsTrained);
    }
}