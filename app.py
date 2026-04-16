from flask import Flask, request, jsonify, send_from_directory
import requests
import re
import json
import os
import time

app = Flask(__name__)
session = requests.Session()

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Get the directory where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

SYSTEM_INSTRUCTION = """You are an elite product strategy AI agent specialising in Kotler's marketing frameworks.
Your task is to thoroughly research the provided product using Google Search.

Search for: features, pricing, competitors, customer reviews, market size, market growth, target audience, positioning, sales data.

Respond ONLY with a valid JSON object matching this schema:
{{
  "product_summary": "one concise line describing what the product is",
  "product_image_url": "valid public image URL of this product (jpg/png), extract directly from search results",
  "category": "industry/category",
  "classification": {{
    "buying_type": "convenience | shopping | specialty | unsought",
    "buying_type_justification": "detailed justification grounded in consumer buying behaviour",
    "durability_tangibility": "nondurable good | durable good | service | hybrid",
    "durability_justification": "detailed justification",
    "overall_classification_summary": "strategic implication for the marketing team"
  }},
  "five_levels": {{
    "core_benefit": {{"content": "fundamental need or want satisfied", "strength": "strong|adequate|weak"}},
    "basic_product": {{"content": "basic physical/digital product features", "strength": "strong|adequate|weak"}},
    "expected_product": {{"content": "what customers normally expect", "strength": "strong|adequate|weak"}},
    "augmented_product": {{"content": "how the product exceeds expectations", "strength": "strong|adequate|weak"}},
    "potential_product": {{"content": "all possible future augmentations", "strength": "strong|adequate|weak"}},
    "gaps_summary": "insightful gap analysis — where over-invested and under-delivered"
  }},
  "differentiation": [
    {{
      "dimension": "Form | Features | Performance Quality | Reliability | Style | Design | Delivery | etc.",
      "title": "short compelling title",
      "recommendation": "specific, concrete, actionable recommendation",
      "justification": "why this follows from the classification and level analysis"
    }},
    {{"dimension": "...", "title": "...", "recommendation": "...", "justification": "..."}},
    {{"dimension": "...", "title": "...", "recommendation": "...", "justification": "..."}}
  ],
  "market_data": {{
    "market_size": "current total addressable market size with source e.g. $4.2B (2024)",
    "market_size_url": "url to verify market size",
    "market_growth_rate": "CAGR percentage e.g. 12.4%",
    "growth_rate_url": "url to verify CAGR",
    "growth_labels": ["2021", "2022", "2023", "2024", "2025E"],
    "growth_values": [100, 115, 132, 148, 168],
    "product_market_share": "estimated share e.g. 8%",
    "market_share_url": "url to verify market share",
    "market_share_value": 8,
    "target_segment": "primary target customer segment description",
    "market_summary": "2-3 sentence summary of the market landscape"
  }},
  "competitors": [
    {{
      "name": "Distinct competing brand & product (NO different retailers of the same product)",
      "price": "price range e.g. Rs.2,999",
      "price_value": 2999,
      "source_url": "url to verify competitor price (ensure this is a unique URL for this specific competitor)",
      "market_share": "e.g. 22%",
      "market_share_value": 22,
      "key_strength": "their main competitive advantage",
      "key_weakness": "their main weakness",
      "threat_level": "high | medium | low"
    }}
  ],
  "sentiment": {{
    "overall_score": 72,
    "positive": 68,
    "neutral": 18,
    "negative": 14,
    "top_positives": ["what customers love 1", "what customers love 2", "what customers love 3"],
    "top_negatives": ["main complaint 1", "main complaint 2", "main complaint 3"],
    "review_summary": "2-3 sentence summary of overall customer sentiment"
  }},
  "pricing": {{
    "product_price": "actual price e.g. Rs.1,299",
    "product_price_url": "url to verify product price",
    "product_price_value": 1299,
    "price_position": "budget | mid-range | premium | luxury",
    "price_summary": "analysis of pricing strategy and value perception",
    "price_comparison": [
      {{"label": "Product name", "value": 1299, "is_subject": true}},
      {{"label": "Competitor 1", "value": 999, "is_subject": false}},
      {{"label": "Competitor 2", "value": 1599, "is_subject": false}},
      {{"label": "Competitor 3", "value": 2499, "is_subject": false}}
    ]
  }}
}}

Use real researched data wherever possible. For growth_values use index 100 as base year and show relative growth. Include 3-5 competitors. Be specific and accurate."""

PRICING_INSTRUCTION = """You are an elite pricing intelligence AI agent.
Your task is to thoroughly research the pricing of the provided product using Google Search.
If previous product strategy data is provided as context, use it to inform your pricing analysis.

Search for: official prices, competitor prices, recent price changes, customer complaints about price, regional prices, and discounts/promotions.

Respond ONLY with a valid JSON object matching this schema:
{
  "product_image_url": "valid public image URL of this product (jpg/png), extract directly from search results",
  "pricing_intelligence": {
    "task1_landscape": {
      "summary": "1-2 paragraph structured summary of the pricing landscape including named competitors, price range, and notable patterns",
      "competitors": [
        {"name": "Distinct competitor brand/product 1 (no retailers)", "price": "e.g. Rs.1,999", "price_value": 1999, "source_url": "explicit distinct URL for this competitor's price"},
        {"name": "Distinct competitor brand/product 2 (no retailers)", "price": "e.g. Rs.2,499", "price_value": 2499, "source_url": "explicit distinct URL for this competitor's price"},
        {"name": "Distinct competitor brand/product 3 (no retailers)", "price": "e.g. Rs.1,799", "price_value": 1799, "source_url": "explicit distinct URL for this competitor's price"}
      ],
      "category_price_range": "e.g. Rs.1,500 - Rs.3,000",
      "price_range_url": "url verifying this range"
    },
    "task2_framework": {
      "step1_objective": {
        "objective": "partial cost recovery | profit margin maximization | revenue maximization | quality leadership | quantity maximization | status quo | survival",
        "justification": "evidence from research",
        "detailed_explanation": "a comprehensive 3-5 sentence description of why this objective is appropriate"
      },
      "step2_demand": {
        "elasticity": "highly elastic | relatively inelastic",
        "reasoning": "signals from research supporting assessment",
        "detailed_explanation": "a comprehensive 3-5 sentence description of why this demand constraint holds true"
      },
      "step3_costs": {
        "variable_costs": "high | low",
        "fixed_costs": "high | low",
        "implications": "what this implies for pricing flexibility",
        "detailed_explanation": "a comprehensive 3-5 sentence description justifying the cost baseline"
      },
      "step4_competitors": {
        "positioning": "above | below | in line",
        "comparison": "comparing to named competitors",
        "detailed_explanation": "a comprehensive 3-5 sentence description of why this positioning is validated"
      },
      "step5_method": {
        "method": "competitive | good-better-best | loss-leader | optional-product | penetration | premium | product-bundle | skim | markup | target-return",
        "justification": "why this method fits best",
        "detailed_explanation": "a comprehensive 3-5 sentence description exploring why this pricing method makes the most economic sense"
      },
      "step6_final_price": {
        "recommendation": "specific final price or price range (must use numbers)",
        "justification": "1 paragraph justification tying the previous 5 steps together",
        "detailed_explanation": "a comprehensive 3-5 sentence explanation confirming why the final price mathematically and strategically fits"
      }
    },
    "task3_biases": {
      "anchoring_bias": {"present": true, "explanation": "brief overview of usage or absence", "detailed_explanation": "a detailed 2-4 sentence description explaining why this bias presence/absence is correct based on evidence", "source_url": "URL validating this bias/strategy"},
      "decoy_effect": {"present": false, "explanation": "brief overview of usage or absence", "detailed_explanation": "a detailed 2-4 sentence description explaining why this bias presence/absence is correct based on evidence", "source_url": "URL validating this bias/strategy"},
      "loss_aversion": {"present": true, "explanation": "brief overview of usage or absence", "detailed_explanation": "a detailed 2-4 sentence description explaining why this bias presence/absence is correct based on evidence", "source_url": "URL validating this bias/strategy"}
    },
    "task4_moves": [
      {
        "move": "concrete, actionable move with specific numbers (e.g. Raise entry tier from Rs.1,999 to Rs.2,199)",
        "justification": "why this makes sense strategically",
        "impact": "expected impact on revenue/perception"
      },
      { "move": "...", "justification": "...", "impact": "..." },
      { "move": "...", "justification": "...", "impact": "..." }
    ]
  }
}

Use real researched data. Provide specific numbers and named competitors. Do not guess confidently if no data exists; instead, state honest uncertainty.
"""

def extract_json_from_gemini(raw_text):
    start_idx = raw_text.find('{')
    end_idx = raw_text.rfind('}')
    if start_idx != -1 and end_idx != -1:
        return raw_text[start_idx:end_idx+1]
    return raw_text

def post_gemini_with_retry(api_key, json_payload, timeout=120, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = session.post(
                f"{GEMINI_API_URL}?key={api_key}",
                json=json_payload,
                timeout=timeout
            )
            if not response.ok:
                err = response.json()
                msg = err.get("error", {}).get("message", "Gemini API error")
                if response.status_code in [429, 503, 500] and attempt < max_retries - 1:
                    time.sleep(10 + (attempt * 5))
                    continue
                raise Exception(msg)
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(10 + (attempt * 5))
                continue
            raise Exception(f"API connection error: {str(e)}")

def call_gemini(api_key, instruction, prompt):
    json_payload = {
        "systemInstruction": {"parts": [{"text": instruction}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 6000
        }
    }
    result = post_gemini_with_retry(api_key, json_payload)
    
    raw_text = ""
    for part in result.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        if "text" in part:
            raw_text += part["text"]
            
    sources = []
    chunks = result.get("candidates", [{}])[0].get("groundingMetadata", {}).get("groundingChunks", [])
    for chunk in chunks:
        if "web" in chunk:
            web_data = chunk["web"]
            if "uri" in web_data and web_data["uri"].startswith("/"):
                web_data["uri"] = f"https://www.google.com{web_data['uri']}"
            sources.append(web_data)
            
    if not raw_text:
        raise Exception("No response from Gemini.")
        
    parsed_json = json.loads(extract_json_from_gemini(raw_text))
    return parsed_json, sources



@app.route("/analyse", methods=["POST"])
def analyse():
    data = request.json
    product = data.get("product", "")
    api_key = data.get("api_key", "")
    analysis_type = data.get("analysis_type", "strategy") # 'strategy', 'pricing', 'both'

    if not product or not api_key:
        return jsonify({"error": "Missing product or API key"}), 400

    try:
        final_result = {}
        all_sources = []
        
        if analysis_type in ["strategy", "both"]:
            strategy_json, sources = call_gemini(api_key, SYSTEM_INSTRUCTION, f"Research the product '{product}' and its top competitors thoroughly using Google Search. FIND unique sources for each competitor.")
            final_result.update(strategy_json)
            all_sources.extend(sources)
            
        if analysis_type in ["pricing", "both"]:
            if analysis_type == "both":
                time.sleep(3) # Give the API a brief rest to avoid concurrency rate limits
            prompt = f"Research the pricing of the product '{product}' AND its active competitors thoroughly using Google Search. ENSURE you retrieve distinct, unique URLs for each competitor you list."
            if analysis_type == "both":
                prompt += f"\n\nHere is the Product Strategy context to use logically:\n{json.dumps(final_result)}"
            
            pricing_json, sources = call_gemini(api_key, PRICING_INSTRUCTION, prompt)
            final_result.update(pricing_json)
            all_sources.extend(sources)

        final_result["sources"] = all_sources
        final_result["product_name"] = product
        final_result["analysis_type_run"] = analysis_type

        return jsonify(final_result)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API connection error: {str(e)}"}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "Could not parse AI response. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare", methods=["POST"])
def compare():
    data = request.json
    p1 = data.get("product1", "")
    p2 = data.get("product2", "")
    api_key = data.get("api_key", "")

    if not p1 or not p2 or not api_key:
        return jsonify({"error": "Missing product or API key"}), 400

    base_schema = SYSTEM_INSTRUCTION[SYSTEM_INSTRUCTION.find('{'):SYSTEM_INSTRUCTION.rfind('}')+1]
    
    comp_inst = f"""You are an elite product strategy AI agent.
Your task is to thoroughly research TWO provided products using Google Search.
Respond ONLY with a valid JSON object matching this schema:
{{
  "product_a": {base_schema},
  "product_b": {base_schema}
}}
Use real researched data wherever possible."""

    try:
        json_payload = {
            "systemInstruction": {"parts": [{"text": comp_inst}]},
            "contents": [{"parts": [{"text": f"Research products '{p1}' and '{p2}' thoroughly using Google Search."}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192}
        }
        res_json = post_gemini_with_retry(api_key, json_payload, timeout=150)

        raw_text = ""
        for part in res_json.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if "text" in part:
                raw_text += part["text"]

        if not raw_text:
            return jsonify({"error": "No response from Gemini."}), 500

        sources = []
        chunks = res_json.get("candidates", [{}])[0].get("groundingMetadata", {}).get("groundingChunks", [])
        for chunk in chunks:
            if "web" in chunk:
                sources.append(chunk["web"])

        # Clean response string to extract valid JSON
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            raw_text = raw_text[start_idx:end_idx+1]
            
        parsed = json.loads(raw_text)
        
        if "product_a" in parsed:
            parsed["product_a"]["sources"] = sources
            parsed["product_a"]["product_name"] = p1
        if "product_b" in parsed:
            parsed["product_b"]["sources"] = sources
            parsed["product_b"]["product_name"] = p2

        return jsonify(parsed)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("")
    print("================================================")
    print("  Product Strategy Agent is running!")
    print("  Open your browser and go to:")
    print("  http://localhost:5000")
    print("================================================")
    print("")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
