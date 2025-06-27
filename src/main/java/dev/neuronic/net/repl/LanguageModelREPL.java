package dev.neuronic.net.repl;

import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.SamplingConfig;
import dev.neuronic.net.repl.transformers.TokenizerTransformer;
import java.util.*;

/**
 * Specialized REPL for language models with text generation features.
 * Provides an easy-to-use interface for interactive text generation.
 */
public class LanguageModelREPL {
    
    private final SimpleNetLanguageModel model;
    private final int maxResponseLength;
    private final Set<String> sentenceEnders;
    
    /**
     * Create a language model REPL with custom settings.
     */
    public LanguageModelREPL(SimpleNetLanguageModel model, int maxResponseLength) {
        this.model = model;
        this.maxResponseLength = maxResponseLength;
        this.sentenceEnders = Set.of(".", "!", "?");
    }
    
    /**
     * Start an interactive REPL session with default settings.
     */
    public static void start(SimpleNetLanguageModel model) {
        new LanguageModelREPL(model, 100).run();
    }
    
    /**
     * Start an interactive REPL session with custom max response length.
     */
    public static void start(SimpleNetLanguageModel model, int maxResponseLength) {
        new LanguageModelREPL(model, maxResponseLength).run();
    }
    
    /**
     * Run the interactive REPL.
     */
    public void run() {
        // Create the generic REPL with language model specific configuration
        SimpleNetREPL<String[], String> repl = new SimpleNetREPL<>(
            model,
            TokenizerTransformer.splitOnSpaces(),
            this::formatOutput,
            this::generateResponse,
            this::predictNext
        );
        
        repl.withPrompt("> ")
            .withWelcomeMessage(
                "=== Language Model REPL ===\n" +
                "Enter text to get AI responses. The model will generate complete sentences.\n" +
                "Current sampling: Temperature 0.8 (adjustable in code)\n" +
                "Commands:\n" +
                "  generate:<text> - Force generation mode\n" +
                "  quit - Exit\n\n"
            )
            .run();
    }
    
    private String predictNext(Object model, String[] input) {
        SimpleNetLanguageModel lm = (SimpleNetLanguageModel) model;
        
        // Always generate a full response for better user experience
        return generateResponse(model, input, maxResponseLength);
    }
    
    private String generateResponse(Object model, String[] input, int maxSteps) {
        SimpleNetLanguageModel lm = (SimpleNetLanguageModel) model;
        List<String> context = new ArrayList<>(Arrays.asList(input));
        StringBuilder response = new StringBuilder();
        
        // Set temperature for more diverse generation
        SamplingConfig oldConfig = lm.getSamplingConfig();
        lm.setSamplingConfig(SamplingConfig.temperature(0.8f));
        
        try {
            for (int i = 0; i < maxSteps; i++) {
                // Get context window
                String[] contextArray = context.toArray(new String[0]);
                
                // Predict next word
                String[] paddedContext = lm.padSequence(contextArray);
                String nextWord = lm.predictNext(paddedContext);
                
                // Add to response
                if (i > 0 && !sentenceEnders.contains(nextWord)) {
                    response.append(" ");
                }
                response.append(nextWord);
                
                // Update context
                context.add(nextWord);
                
                // Stop at sentence end
                if (sentenceEnders.contains(nextWord)) {
                    break;
                }
            }
        } finally {
            // Restore original config
            lm.setSamplingConfig(oldConfig);
        }
        
        return response.toString();
    }
    
    private String formatOutput(String output) {
        return "Response: " + output;
    }
}