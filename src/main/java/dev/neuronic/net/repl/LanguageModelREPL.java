package dev.neuronic.net.repl;

import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.SamplingConfig;
import dev.neuronic.net.SamplingStrategies;
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
    private final boolean avoidUnk;
    
    /**
     * Create a language model REPL with custom settings.
     * 
     * @param model the language model to use
     * @param maxResponseLength maximum tokens to generate
     * @param sentenceEnders custom sentence enders (e.g., ".", "!", "?", "\n")
     * @param avoidUnk if true, skip UNK tokens and use next best predictions
     */
    public LanguageModelREPL(SimpleNetLanguageModel model, int maxResponseLength, 
                            Set<String> sentenceEnders, boolean avoidUnk) {
        this.model = model;
        this.maxResponseLength = maxResponseLength;
        this.sentenceEnders = sentenceEnders != null ? sentenceEnders : Set.of(".", "!", "?");
        this.avoidUnk = avoidUnk;
    }
    
    
    /**
     * Start an interactive REPL session with default settings.
     * 
     * @param model the language model
     * @param avoidUnk if true, skip UNK tokens and use next best predictions
     */
    public static void start(SimpleNetLanguageModel model, boolean avoidUnk) {
        new LanguageModelREPL(model, 100, null, avoidUnk).run();
    }
    
    /**
     * Start an interactive REPL session with UNK avoidance.
     * 
     * <p><b>Example - Avoid unknown tokens:</b>
     * <pre>{@code
     * // Generate text but skip any UNK tokens
     * LanguageModelREPL.start(model, 200, true);
     * }</pre>
     * 
     * @param model the language model
     * @param maxResponseLength maximum tokens to generate  
     * @param avoidUnk if true, skip UNK tokens and use next best predictions
     */
    public static void start(SimpleNetLanguageModel model, int maxResponseLength, boolean avoidUnk) {
        new LanguageModelREPL(model, maxResponseLength, null, avoidUnk).run();
    }
    
    /**
     * Start an interactive REPL session with custom settings and UNK avoidance.
     * 
     * <p><b>Example - Custom enders with UNK avoidance:</b>
     * <pre>{@code
     * // For code generation, stop at newlines/semicolons and avoid UNK
     * LanguageModelREPL.start(
     *     model, 
     *     200,  // max tokens
     *     Set.of(";", "\n", "{", "}"),
     *     true  // avoid UNK tokens
     * );
     * }</pre>
     * 
     * @param model the language model
     * @param maxResponseLength maximum tokens to generate
     * @param sentenceEnders tokens that end a response
     * @param avoidUnk if true, skip UNK tokens and use next best predictions
     */
    public static void start(SimpleNetLanguageModel model, int maxResponseLength,
                           Set<String> sentenceEnders, boolean avoidUnk) {
        new LanguageModelREPL(model, maxResponseLength, sentenceEnders, avoidUnk).run();
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

    private boolean endsWithSentenceEnder(String word) {
        return sentenceEnders.stream().anyMatch(word::endsWith);
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
                String nextWord;
                
                if (avoidUnk) {
                    // Get the UNK token ID to exclude
                    int unkId = lm.getTokenId("<unk>");
                    int[] excludeIndices = new int[]{unkId};
                    
                    // Get probabilities for the padded context
                    float[] probs = lm.predictProbabilities(paddedContext);
                    
                    // Get current sampling config settings
                    SamplingConfig config = lm.getSamplingConfig();
                    
                    // Sample with the current sampling config and exclusions
                    int sampledTokenId = switch (config.getStrategy()) {
                        case ARGMAX -> {
                            // Find argmax excluding UNK
                            int bestIdx = -1;
                            float bestProb = -1;
                            for (int j = 0; j < probs.length; j++) {
                                if (j != unkId && probs[j] > bestProb) {
                                    bestProb = probs[j];
                                    bestIdx = j;
                                }
                            }
                            yield bestIdx;
                        }
                        case TEMPERATURE -> SamplingStrategies.sampleWithTemperature(
                            probs, config.getTemperature(), excludeIndices, 
                            lm.getUnderlyingNet().getRandom()
                        );
                        case TOP_K -> SamplingStrategies.sampleTopK(
                            probs, config.getK(), config.getTemperature(),
                            excludeIndices, lm.getUnderlyingNet().getRandom()
                        );
                        case TOP_P -> SamplingStrategies.sampleTopP(
                            probs, config.getP(), config.getTemperature(),
                            excludeIndices, lm.getUnderlyingNet().getRandom()
                        );
                    };
                    
                    nextWord = lm.getWord(sampledTokenId);
                } else {
                    nextWord = lm.predictNext(paddedContext);
                }
                
                // Add to response
                if (i > 0 && !sentenceEnders.contains(nextWord)) {
                    response.append(" ");
                }
                response.append(nextWord);
                
                // Update context
                context.add(nextWord);
                
                // Stop at sentence end
                if (endsWithSentenceEnder(nextWord))
                    break;
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