# Backend Calculations & Models Documentation

## Overview
This document covers all calculations, models, and processing modes used in the UserChurn backend system. The system supports three processing modes: Rule-based, LLM-based, and Hybrid.

## Processing Modes

### 1. Rule-Based Mode
- **Pattern Detection**: Embedding-based similarity matching
- **Sentiment Analysis**: VADER + Advanced Transformers
- **Emotion Analysis**: HuggingFace API + Basic keyword matching
- **Offer Filtering**: Rule-based logic with conversation context

### 2. LLM-Based Mode
- **Pattern Detection**: LLM indicator extraction
- **Sentiment Analysis**: LLM-based analysis
- **Emotion Analysis**: LLM-based analysis
- **Offer Filtering**: LLM-based intelligent filtering

### 3. Hybrid Mode
- **Pattern Detection**: Rule-based with higher thresholds
- **Sentiment Analysis**: Rule-based with higher thresholds
- **Emotion Analysis**: Rule-based with higher thresholds
- **Offer Filtering**: Rule-based with enhanced context

## Core Calculations

### Churn Score Calculation
**Formula**: Exponential Moving Average
```
new_score = (1 - α) × previous_score + α × (previous_score + total_risk_delta)
```

**Parameters**:
- `α = 0.3` (exponential moving average factor)
- `baseline_score = 50.0` (starting churn risk 0-100)
- `total_risk_delta = pattern_risk + sentiment_risk + emotion_risk`

**Score Bounds**: 0.0 to 100.0

### Risk Pattern Detection

#### Rule-Based Pattern Detection
**Model**: `all-mpnet-base-v2` (SentenceTransformer)
**Method**: Cosine similarity between customer text and pattern examples

**Similarity Calculation**:
```
similarity = dot(pattern_embeddings, text_embedding) / (norm(pattern_embeddings) × norm(text_embedding))
```

**Thresholds**:
- **Rule-based**: 0.45
- **Hybrid**: 0.6

**Risk Scaling**:
```
similarity_factor = (max_similarity - threshold) / (1.0 - threshold)
risk_contribution = base_risk × similarity_factor
```

**Pattern Risk Values**:
- `billing_complaint`: 15.0
- `competitor_mention`: 30.0
- `service_frustration`: 20.0
- `process_frustration`: 20.0
- `positive_resolution`: -40.0

#### LLM Pattern Detection
**Model**: `meta-llama/Meta-Llama-3-8B-Instruct` (HuggingFace)
**Confidence**: 0.7 (assumed LLM confidence)
**Threshold**: 0.45 (same as rule-based)
**Calculation**: Same similarity factor logic as rule-based

### Sentiment Analysis

#### Rule-Based Sentiment
**Primary Model**: VADER SentimentIntensityAnalyzer
**Secondary Model**: `distilbert-base-uncased-finetuned-sst-2-english`

**Combined Calculation**:
```
final_sentiment = 0.5 × vader_sentiment + 0.5 × advanced_sentiment
final_confidence = 0.5 × vader_confidence + 0.5 × advanced_confidence
```

**Sentiment Risk Mapping**:
- `sentiment_score < -0.6`: 30.0 (very negative)
- `sentiment_score < -0.3`: 20.0 (moderately negative)
- `sentiment_score < -0.1`: 10.0 (slightly negative)
- `sentiment_score > 0.36`: -20.0 (positive)
- `sentiment_score > 0.25`: -10.0 (slightly positive)

#### LLM Sentiment
**Conversion Mapping**:
- `very_positive`: 0.5
- `positive`: 0.3
- `neutral`: 0.0
- `negative`: -0.2
- `very_negative`: -0.7

### Emotion Analysis

#### Rule-Based Emotion
**Primary**: HuggingFace API (`j-hartmann/emotion-english-distilroberta-base`)
**Fallback**: Basic keyword matching

**Keyword-Based Scoring**:
```
score = sum(keyword_matches) / total_keywords
emotion_score = min(score × 3, 1.0)
```

**Emotion Risk Values**:
- `anger`: 25.0
- `disgust`: 20.0
- `fear`: 10.0
- `joy`: -30.0
- `neutral`: 0.0
- `sadness`: 10.0
- `surprise`: 5.0
- `frustration`: 10.0

**Hybrid Mode Threshold**: 0.7
**Risk Scaling** (Hybrid only):
```
similarity_factor = (dominant_score - 0.7) / (1.0 - 0.7)
emotion_risk = base_risk × similarity_factor
```

#### LLM Emotion
**Model**: Same as LLM sentiment analysis
**Output**: Direct emotion classification

## Offer Filtering Logic

### Rule-Based Filtering
**Criteria**:
1. **Price Ceiling**: Remove offers > `conversation_context.price_ceiling`
2. **Unused Services**: Remove offers containing services in `conversation_context.unused_services`
3. **Sorting Logic**:
   - Competitor mentions → Sort by price (ascending)
   - Price concerns → Sort by price (ascending)
   - Default → Sort by relevance score (descending)

### LLM-Based Filtering
**Multi-Step Process**:

#### Step 1: Service Removal Interest
- Remove offers with services customer wants to remove
- Check: `tv_removal_interest`, `mobile_removal_interest`, `internet_removal_interest`

#### Step 2: Budget Filtering
- **High concern**: Remove offers > `customer_mrc`
- **Medium concern**: Keep offers < `customer_mrc`, sort high-to-low
- **None**: No filtering

#### Step 3: Service Usage
- **Low usage**: Move offers to bottom (deprioritize)
- **Normal usage**: Keep in main offers

#### Step 4: Competitor Mention Sorting
- Sort by lowest price first (retention focus)

#### Step 5: Contract Type Preferences
- Remove long-term contracts if `contract_flexibility_needed = True`

#### Step 6: Value Preferences
- **Price-focused**: Sort by lowest price
- **Feature-focused**: Sort by priority (lower number = higher features)
- **Balanced**: No additional sorting

### Offer Priority System
**Priority 1 = Most Features**:
1. SmartHome bundles (Priority 1)
2. Premium Voice + Mobile (Priority 2)
3. Premium TV bundles (Priority 3)
4. High-speed internet (Priority 4-5)
5. Standard bundles (Priority 6-7)
6. Basic bundles (Priority 8-11)

## Context Tracking

### Conversation Context
```python
@dataclass
class ConversationContext:
    mentioned_prices: List[float]
    unused_services: List[str]
    price_trend: str  # 'increasing', 'stable', 'decreasing'
    patterns_detected: List[str]
    sentiment_trend: str  # 'positive', 'negative', 'neutral'
    competitor_mentions: List[str]
    current_bill: Optional[float] = None
    price_ceiling: Optional[float] = None
```

### Price Extraction
**Patterns**:
- `\$(\d+(?:\.\d{2})?)` - $200, $180.50
- `(\d+) dollars?` - 200 dollars
- `paying (\d+)` - paying 200
- Written numbers: "two hundred dollars" → 200

### Service Usage Detection
**TV Unused**: "barely watch tv", "don't watch tv", "never use tv"
**Mobile Unused**: "don't use mobile", "rarely call", "never use phone"
**Internet Unused**: "slow internet", "don't need fast", "basic internet"

### Competitor Detection
**Competitors**: tmobile, t-mobile, verizon, att, at&t, comcast, spectrum, xfinity, cox, charter, directv, dish

## Customer Profile
```python
customer_profile = {
    "name": "Megan Hazelwood",
    "current_mrc": 200,
    "previous_mrc": 180,
    "tenure_months": 18,
    "current_plan": "$200Data_TV_MOB_200MBPkg_2yr",
    "services": ["Internet", "TV", "Mobile"]
}
```

## Offer Catalog Structure
```python
{
    "offer_id": "BB+PKG_$240TVplus_MOB",
    "title": "BB+PKG_$240TVplus_MOB",
    "description": "TV Bundle with Premium Channels\nInternet, TV, Mobile",
    "value_proposition": "Premium entertainment package",
    "price_delta": 240,
    "product_types": ["Internet", "TV", "Mobile"],
    "contract_type": "standard",
    "retention_offer": False,
    "category": "bundle",
    "priority": 3
}
```

## Processing Mode Configuration

### Mode Switching
```python
# LLM Mode
scorer.use_llm_indicators = True
scorer.use_llm_offer_filtering = True
scorer.use_hybrid_processing = False

# Hybrid Mode
scorer.use_llm_indicators = False
scorer.use_llm_offer_filtering = False
scorer.use_hybrid_processing = True

# Rule-based Mode
scorer.use_llm_indicators = False
scorer.use_llm_offer_filtering = False
scorer.use_hybrid_processing = False
```

### Threshold Differences
| Component | Rule-based | Hybrid | LLM |
|-----------|------------|--------|-----|
| Pattern Similarity | 0.45 | 0.6 | 0.45 |
| Emotion Score | No threshold | 0.7 | No threshold |
| Processing | Embedding-based | Embedding-based | LLM-based |

## API Dependencies

### Required Models
- **VADER**: `vaderSentiment`
- **Embeddings**: `sentence-transformers` (all-mpnet-base-v2)
- **Advanced Sentiment**: `transformers` (distilbert-base-uncased-finetuned-sst-2-english)
- **Emotion**: HuggingFace API (j-hartmann/emotion-english-distilroberta-base)
- **LLM**: HuggingFace API (meta-llama/Meta-Llama-3-8B-Instruct)

### Environment Variables
- `HUGGINGFACEHUB_API_TOKEN`: Required for LLM and emotion analysis

## Error Handling & Fallbacks

### LLM Failures
- **Pattern Detection**: Falls back to embedding-based detection
- **Offer Filtering**: Falls back to rule-based filtering
- **Analysis**: Returns error object with fallback data

### Model Failures
- **Advanced Sentiment**: Falls back to VADER only
- **Emotion API**: Falls back to keyword-based detection
- **Embeddings**: Required, no fallback

## Performance Considerations

### LLM Processing
- **Batch Processing**: Accumulates messages for efficient processing
- **Context Window**: Maintains conversation context across calls
- **Caching**: Reuses analysis results to avoid duplicate API calls

### Memory Management
- **Conversation History**: Stores all churn events
- **Risk Events**: Tracks score changes over time
- **Previous Offers**: Limits to last 10 offers to prevent memory bloat

## Output Formats

### Churn Event
```python
@dataclass
class ChurnEvent:
    timestamp: datetime
    speaker: str
    text: str
    agent_context: str
    sentiment_score: float
    emotion_result: dict
    risk_delta: float
    cumulative_score: float
    confidence: float
    detected_patterns: list
```

### Offer Response
```python
{
    'id': 'offer_id',
    'title': 'offer_title',
    'description': 'offer_description',
    'value': '$240/month',
    'urgency': 'high',
    'category': 'bundle',
    'relevance': 90,
    'accepted': True,
    'rejection_reason': None,
    'llm_filtered': True
}
```

This documentation covers all essential calculations, models, and processing logic used in the UserChurn backend system across all three processing modes.
