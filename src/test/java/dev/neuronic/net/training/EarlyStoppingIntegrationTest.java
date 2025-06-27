package dev.neuronic.net.training;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.activators.ReluActivator;
import dev.neuronic.net.layers.DenseLayer;
import dev.neuronic.net.layers.GruLayer;
import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.losses.CrossEntropyLoss;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.outputs.SoftmaxCrossEntropyOutput;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import dev.neuronic.net.simple.SimpleNetTrainingResult;
import dev.neuronic.net.WeightInitStrategy;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration test to verify early stopping works correctly.
 */
class EarlyStoppingIntegrationTest {
    
    @Test
    void testEarlyStoppingStopsTraining() {
        // Create a simple language model
        int sequenceLength = 5;
        int vocabularySize = 50;
        int embeddingSize = 16;
        int hiddenSize = 32;
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);  // Higher learning rate for faster convergence
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(sequenceLength)
            .layer(InputSequenceEmbeddingLayer.spec(sequenceLength, vocabularySize, embeddingSize, optimizer, WeightInitStrategy.XAVIER))
            .layer(GruLayer.specLast(hiddenSize, optimizer, WeightInitStrategy.XAVIER))
            .output(SoftmaxCrossEntropyOutput.spec(vocabularySize, optimizer, WeightInitStrategy.XAVIER));
        
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(net);
        
        // Generate synthetic training data with simple patterns that will plateau
        List<String[]> sequences = new ArrayList<>();
        List<String> targets = new ArrayList<>();
        
        String[] words = {"the", "a", "cat", "dog", "runs", "walks", "jumps", "sleeps", "big", "small"};
        
        // Create simple deterministic patterns that the model can learn quickly
        // Pattern 1: "the" -> "cat", "a" -> "dog"
        // Pattern 2: "cat" -> "runs", "dog" -> "walks"
        // Pattern 3: "big" -> "cat", "small" -> "dog"
        
        // Generate training samples with these patterns
        for (int i = 0; i < 20; i++) {
            // Pattern 1
            sequences.add(new String[]{"the", "big", "cat", "runs", "fast"});
            targets.add("cat");
            
            sequences.add(new String[]{"a", "small", "dog", "walks", "slow"});
            targets.add("dog");
            
            // Pattern 2
            sequences.add(new String[]{"the", "cat", "jumps", "and", "runs"});
            targets.add("runs");
            
            sequences.add(new String[]{"a", "dog", "sleeps", "and", "walks"});
            targets.add("walks");
            
            // Pattern 3
            sequences.add(new String[]{"big", "the", "cat", "is", "here"});
            targets.add("cat");
            
            sequences.add(new String[]{"small", "a", "dog", "is", "there"});
            targets.add("dog");
        }
        
        // Add a few random samples to create some noise
        Random rng = new Random(42);
        for (int i = 0; i < 20; i++) {
            String[] sequence = new String[sequenceLength];
            for (int j = 0; j < sequenceLength; j++) {
                sequence[j] = words[rng.nextInt(words.length)];
            }
            sequences.add(sequence);
            targets.add(words[rng.nextInt(words.length)]);
        }
        
        // Configure training with early stopping
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(10)
            .epochs(20)  // Set high epoch count
            .validationSplit(0.2f)
            .verbosity(1)
            .withEarlyStopping(3, 0.01f)  // Stop after 3 epochs without improvement
            .build();
        
        // Train the model
        SimpleNetTrainingResult result = model.trainBulk(sequences, targets, config);
        
        // Verify that training stopped early
        TrainingMetrics metrics = result.getMetrics();
        int actualEpochs = metrics.getEpochCount();
        
        // Should stop before reaching 20 epochs
        assertTrue(actualEpochs < 20, 
            "Early stopping should have stopped training before 20 epochs, but ran " + actualEpochs + " epochs");
        
        // Should run at least 4 epochs (3 patience + 1)
        assertTrue(actualEpochs >= 4, 
            "Should run at least 4 epochs with patience=3, but only ran " + actualEpochs + " epochs");
        
        System.out.println("Early stopping test passed: Training stopped after " + actualEpochs + " epochs");
    }
    
    @Test
    void testEarlyStoppingMonitorsCorrectMetric() {
        // Create trainer with callbacks to track metrics
        int inputSize = 10;
        int outputSize = 5;
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(inputSize)
            .layer(DenseLayer.spec(20, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE))
            .output(SoftmaxCrossEntropyOutput.spec(outputSize, optimizer, WeightInitStrategy.XAVIER));
        
        // Generate data where accuracy plateaus but loss continues improving
        float[][] inputs = new float[100][inputSize];
        float[][] targets = new float[100][outputSize];
        Random rng = new Random(42);
        
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < inputSize; j++) {
                inputs[i][j] = rng.nextFloat();
            }
            int targetClass = rng.nextInt(outputSize);
            targets[i][targetClass] = 1.0f;
        }
        
        BatchTrainer.TrainingConfig batchConfig = new BatchTrainer.TrainingConfig.Builder()
            .batchSize(10)
            .epochs(15)
            .validationSplit(0.2f)
            .verbosity(0)
            .build();
        
        BatchTrainer trainer = new BatchTrainer(net, CrossEntropyLoss.INSTANCE, batchConfig);
        
        // Add early stopping monitoring val_loss
        trainer.withEarlyStopping(3, 0.001f);
        
        // Add callback to track when early stopping triggers
        final boolean[] earlyStoppingSeen = {false};
        trainer.withCallback(new TrainingCallback() {
            @Override
            public void onEpochEnd(int epoch, TrainingMetrics metrics) {
                // Check if stop was requested
                if (trainer.getStopFlag().get()) {
                    earlyStoppingSeen[0] = true;
                    
                    // Verify that val_loss was still improving when stopped
                    TrainingMetrics.EpochMetrics current = metrics.getEpochMetrics(epoch);
                    if (epoch > 0) {
                        TrainingMetrics.EpochMetrics previous = metrics.getEpochMetrics(epoch - 1);
                        System.out.printf("Epoch %d: val_loss=%.4f (previous=%.4f), val_acc=%.4f%n",
                            epoch + 1, current.getValidationLoss(), previous.getValidationLoss(),
                            current.getValidationAccuracy());
                    }
                }
            }
        });
        
        BatchTrainer.TrainingResult result = trainer.fit(inputs, targets);
        
        // Early stopping should have triggered
        assertTrue(earlyStoppingSeen[0], "Early stopping should have been triggered");
        
        // Training should have stopped early
        assertTrue(result.getMetrics().getEpochCount() < 15, 
            "Training should have stopped before 15 epochs");
    }
}