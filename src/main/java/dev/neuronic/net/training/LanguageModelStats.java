package dev.neuronic.net.training;

/**
 * Training statistics specific to language modeling tasks.
 * Provides language model-focused metrics with clear naming and documentation.
 */
public class LanguageModelStats extends TrainingStats {
    
    // Language model specific metrics
    private final double bitsPerCharacter;
    private final double bitsPerWord;
    private final int sequenceLength;
    private final double averageWordLength;
    private final int activeVocabulary;  // Words actually used in predictions
    private final double unkRate;        // Percentage of unknown tokens
    
    // Generation quality indicators
    private final double repetitionRate;
    private final double diversityScore;
    
    private LanguageModelStats(Builder builder) {
        super(builder);
        this.bitsPerCharacter = builder.bitsPerCharacter;
        this.bitsPerWord = builder.bitsPerWord;
        this.sequenceLength = builder.sequenceLength;
        this.averageWordLength = builder.averageWordLength;
        this.activeVocabulary = builder.activeVocabulary;
        this.unkRate = builder.unkRate;
        this.repetitionRate = builder.repetitionRate;
        this.diversityScore = builder.diversityScore;
    }
    
    /**
     * Bits per character (cross-entropy / log(2)).
     * Lower is better. Good models: < 1.5 bits/char.
     */
    public double getBitsPerCharacter() {
        return bitsPerCharacter;
    }
    
    /**
     * Bits per word.
     * Lower is better. Good models: < 8 bits/word.
     */
    public double getBitsPerWord() {
        return bitsPerWord;
    }
    
    /**
     * Context window size (sequence length).
     */
    public int getSequenceLength() {
        return sequenceLength;
    }
    
    /**
     * Average word length in characters.
     */
    public double getAverageWordLength() {
        return averageWordLength;
    }
    
    /**
     * Number of unique words model actively predicts.
     * May be less than vocabulary size if some words are never predicted.
     */
    public int getActiveVocabulary() {
        return activeVocabulary;
    }
    
    /**
     * Percentage of unknown tokens in validation (0-100).
     * High values indicate vocabulary is too small.
     */
    public double getUnkRate() {
        return unkRate;
    }
    
    /**
     * Rate of repetitive predictions (0-1).
     * Lower is better. High values indicate model is stuck in loops.
     */
    public double getRepetitionRate() {
        return repetitionRate;
    }
    
    /**
     * Diversity of predictions (0-1).
     * Higher is better. Low values indicate limited vocabulary usage.
     */
    public double getDiversityScore() {
        return diversityScore;
    }
    
    /**
     * Get perplexity interpretation as human-readable string.
     */
    public String getPerplexityInterpretation() {
        double perplexity = getValidationPerplexity();
        if (perplexity < 20) {
            return "Excellent - Near human-level";
        } else if (perplexity < 50) {
            return "Very Good - Fluent text";
        } else if (perplexity < 100) {
            return "Good - Coherent text";
        } else if (perplexity < 200) {
            return "Fair - Mostly coherent";
        } else if (perplexity < 500) {
            return "Poor - Often incoherent";
        } else {
            return "Very Poor - Random output";
        }
    }
    
    /**
     * Estimate memory requirements for the model in MB.
     */
    public double getEstimatedMemoryMB() {
        // Rough estimate: vocab * embedding_dim * 4 bytes
        // Plus overhead for other layers
        return (getVocabularySize() * 128 * 4) / (1024.0 * 1024.0);
    }
    
    /**
     * Check if vocabulary size is appropriate for the dataset.
     */
    public boolean hasAppropriateVocabulary() {
        return unkRate < 5.0 && getVocabularyCoverage() > 95.0;
    }
    
    @Override
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Language Model Training Summary ===\n");
        sb.append(String.format("Training time: %.1fs\n", getTrainingTimeSeconds()));
        sb.append(String.format("Epochs trained: %d\n", getEpochsTrained()));
        
        // Perplexity (primary metric)
        sb.append(String.format("Final perplexity: %.1f (train) / %.1f (val)\n", 
                              getTrainPerplexity(), getValidationPerplexity()));
        sb.append(String.format("Best perplexity: %.1f (epoch %d)\n", 
                              getBestValidationPerplexity(), getBestEpoch() + 1));
        sb.append(String.format("Quality: %s\n", getPerplexityInterpretation()));
        
        // Vocabulary info
        if (getVocabularySize() > 0) {
            sb.append(String.format("Vocabulary: %,d words (%.1f%% coverage)\n", 
                                  getVocabularySize(), getVocabularyCoverage()));
            if (activeVocabulary > 0 && activeVocabulary < getVocabularySize()) {
                sb.append(String.format("Active vocabulary: %,d words (%.1f%%)\n",
                                      activeVocabulary, 
                                      (activeVocabulary * 100.0 / getVocabularySize())));
            }
        }
        
        // Data quality
        if (!Double.isNaN(unkRate)) {
            sb.append(String.format("Unknown token rate: %.1f%%", unkRate));
            if (unkRate > 5)
                sb.append(" (consider larger vocabulary)");
            sb.append("\n");
        }
        
        // Additional metrics
        if (!Double.isNaN(bitsPerCharacter)) {
            sb.append(String.format("Bits per character: %.2f\n", bitsPerCharacter));
        }
        
        if (sequenceLength > 0) {
            sb.append(String.format("Context window: %d tokens\n", sequenceLength));
        }
        
        // Generation quality
        if (!Double.isNaN(diversityScore)) {
            sb.append(String.format("Diversity score: %.2f", diversityScore));
            if (diversityScore < 0.3)
                sb.append(" (low - consider temperature sampling)");
            sb.append("\n");
        }
        
        // Recommendations
        if (getValidationPerplexity() > 200 && getEpochsTrained() < 10) {
            sb.append("Recommendation: Train for more epochs\n");
        } else if (unkRate > 10) {
            sb.append("Recommendation: Increase vocabulary size\n");
        } else if (getValidationPerplexity() > 500) {
            sb.append("Recommendation: Review model architecture\n");
        }
        
        if (wasEarlyStopped())
            sb.append("Training stopped early due to convergence\n");
        
        return sb.toString();
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder extends TrainingStats.Builder {
        private double bitsPerCharacter = Double.NaN;
        private double bitsPerWord = Double.NaN;
        private int sequenceLength;
        private double averageWordLength = 5.0; // English average
        private int activeVocabulary;
        private double unkRate = Double.NaN;
        private double repetitionRate = Double.NaN;
        private double diversityScore = Double.NaN;
        
        public Builder bitsPerCharacter(double bitsPerCharacter) {
            this.bitsPerCharacter = bitsPerCharacter;
            return this;
        }
        
        public Builder bitsPerWord(double bitsPerWord) {
            this.bitsPerWord = bitsPerWord;
            return this;
        }
        
        public Builder sequenceLength(int sequenceLength) {
            this.sequenceLength = sequenceLength;
            return this;
        }
        
        public Builder averageWordLength(double averageWordLength) {
            this.averageWordLength = averageWordLength;
            return this;
        }
        
        public Builder activeVocabulary(int activeVocabulary) {
            this.activeVocabulary = activeVocabulary;
            return this;
        }
        
        public Builder unkRate(double unkRate) {
            this.unkRate = unkRate;
            return this;
        }
        
        public Builder repetitionRate(double repetitionRate) {
            this.repetitionRate = repetitionRate;
            return this;
        }
        
        public Builder diversityScore(double diversityScore) {
            this.diversityScore = diversityScore;
            return this;
        }
        
        /**
         * Calculate bits per character from loss if not set.
         */
        public Builder autoCalculateBits() {
            if (Double.isNaN(bitsPerCharacter)) {
                // Convert nats to bits: loss / ln(2)
                bitsPerCharacter = validationLoss / Math.log(2);
            }
            if (Double.isNaN(bitsPerWord) && !Double.isNaN(bitsPerCharacter)) {
                bitsPerWord = bitsPerCharacter * averageWordLength;
            }
            return this;
        }
        
        @Override
        public LanguageModelStats build() {
            // Auto-calculate perplexity for language models
            autoCalculatePerplexity();
            autoCalculateBits();
            return new LanguageModelStats(this);
        }
    }
}