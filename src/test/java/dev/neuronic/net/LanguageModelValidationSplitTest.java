package dev.neuronic.net;

import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import dev.neuronic.net.simple.SimpleNetTrainingResult;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to verify the validation split issue in language models.
 */
public class LanguageModelValidationSplitTest {
    
    @Test
    public void testTrainWithoutValidationSplit() {
        // Create the same simple pattern
        List<String[]> sequences = new ArrayList<>();
        List<String> targets = new ArrayList<>();
        
        // Simple pattern: "a a a" → "b", "b b b" → "a"
        for (int i = 0; i < 50; i++) {
            sequences.add(new String[]{"a", "a", "a"});
            targets.add("b");
            sequences.add(new String[]{"b", "b", "b"});
            targets.add("a");
        }
        
        // Build model
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
                .layer(Layers.inputSequenceEmbedding(3, 10, 16))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Train WITHOUT validation split
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(20)
            .epochs(20)
            .shuffle(false)
            .validationSplit(0.0f) // NO VALIDATION SPLIT!
            .verbosity(1)
            .build();
            
        SimpleNetTrainingResult result = model.trainBulk(sequences, targets, config);
        
        float finalLoss = result.getFinalLoss();
        System.out.println("Final loss without validation split: " + finalLoss);
        assertTrue(finalLoss < 0.1f, "Should learn pattern without validation split, loss: " + finalLoss);
        
        // Test predictions
        String pred1 = model.predictNext(new String[]{"a", "a", "a"});
        assertEquals("b", pred1, "Should predict 'b' after 'a a a'");
        
        String pred2 = model.predictNext(new String[]{"b", "b", "b"});
        assertEquals("a", pred2, "Should predict 'a' after 'b b b'");
    }
    
    @Test
    public void testManualTrainValidationSplit() {
        // Create mixed data that ensures both patterns in both sets
        List<String[]> trainSequences = new ArrayList<>();
        List<String> trainTargets = new ArrayList<>();
        List<String[]> valSequences = new ArrayList<>();
        List<String> valTargets = new ArrayList<>();
        
        // Add both patterns to training
        for (int i = 0; i < 40; i++) {
            trainSequences.add(new String[]{"a", "a", "a"});
            trainTargets.add("b");
            trainSequences.add(new String[]{"b", "b", "b"});
            trainTargets.add("a");
        }
        
        // Add both patterns to validation
        for (int i = 0; i < 10; i++) {
            valSequences.add(new String[]{"a", "a", "a"});
            valTargets.add("b");
            valSequences.add(new String[]{"b", "b", "b"});
            valTargets.add("a");
        }
        
        // Build model
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
                .layer(Layers.inputSequenceEmbedding(3, 10, 16))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Train with manual split
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(20)
            .epochs(20)
            .shuffle(true)
            .verbosity(1)
            .build();
            
        SimpleNetTrainingResult result = model.trainBulk(
            trainSequences, trainTargets, valSequences, valTargets, config);
        
        float finalLoss = result.getFinalLoss();
        System.out.println("Final loss with manual split: " + finalLoss);
        assertTrue(finalLoss < 0.1f, "Should learn pattern with proper split, loss: " + finalLoss);
    }
}