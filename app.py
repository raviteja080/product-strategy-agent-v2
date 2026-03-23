from flask import Flask, request, jsonify, send_from_directory
import requests
import re
import json
import os

app = Flask(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


def build_prompt(product):
    return f"""You are an elite product strategy AI agent specialising in Kotler's marketing frameworks.
Research the product "{product}" thoroughly using Google Search.

Search for: features, pricing, competitors, customer reviews, market size, market growth, target audience, positioning, sales data.

Respond ONLY with a valid JSON object — no markdown, no preamble, no extra text:
{{
  "product_summary": "one concise line describing what the product is",
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
    "market_growth_rate": "CAGR percentage e.g. 12.4%",
    "growth_labels": ["2021", "2022", "2023", "2024", "2025E"],
    "growth_values": [100, 115, 132, 148, 168],
    "product_market_share": "estimated share e.g. 8%",
    "market_share_value": 8,
    "target_segment": "primary target customer segment description",
    "market_summary": "2-3 sentence summary of the market landscape"
  }},
  "competitors": [
    {{
      "name": "Competitor name",
      "price": "price range e.g. Rs.2,999",
      "price_value": 2999,
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


@app.route("/analyse", methods=["POST"])
def analyse():
    data = request.json
    product = data.get("product", "")
    api_key = data.get("api_key", "")

    if not product or not api_key:
        return jsonify({"error": "Missing product or API key"}), 400

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={api_key}",
            json={
                "contents": [{"parts": [{"text": build_prompt(product)}]}],
                "tools": [{"google_search": {}}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 6000
                }
            },
            timeout=90
        )

        if not response.ok:
            err = response.json()
            return jsonify({"error": err.get("error", {}).get("message", "Gemini API error")}), 400

        result = response.json()

        raw_text = ""
        for part in result.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if "text" in part:
                raw_text += part["text"]

        if not raw_text:
            return jsonify({"error": "No response from Gemini. Please try again."}), 500

        sources = []
        chunks = result.get("candidates", [{}])[0].get("groundingMetadata", {}).get("groundingChunks", [])
        for chunk in chunks:
            if "web" in chunk:
                sources.append(chunk["web"])

        clean = re.sub(r'```json\n?|```', '', raw_text).strip()
        match = re.search(r'\{[\s\S]*\}', clean)
        parsed = json.loads(match.group(0) if match else clean)
        parsed["sources"] = sources
        parsed["product_name"] = product

        return jsonify(parsed)

    except json.JSONDecodeError:
        return jsonify({"error": "Could not parse AI response. Please try again."}), 500
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
