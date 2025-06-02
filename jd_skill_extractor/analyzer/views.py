from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import json
import os
import re
from collections import Counter
import logging
import requests
import hashlib

# Set up logging
logger = logging.getLogger(__name__)

# Initialize Claude client
def get_claude_client():
    """Initialize Claude API client with proper configuration"""
    try:
        if not hasattr(settings, 'CLAUDE_API_KEY') or not settings.CLAUDE_API_KEY:
            logger.error("No Claude API key configured in settings.CLAUDE_API_KEY")
            return None
        
        # Test connection
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': settings.CLAUDE_API_KEY,
            'anthropic-version': '2023-06-01'
        }
        
        # Simple test call
        test_payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "ping"}]
        }
        
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("Claude API connection successful")
            return headers
        else:
            logger.error(f"Claude API test failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Claude initialization failed: {e}")
        return None

# Initialize at module level
claude_headers = get_claude_client()

def index(request):
    return render(request, 'index.html')

def advanced_clean_jd_text(text):
    """Advanced cleaning of job description text with comprehensive preprocessing"""
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase for processing
    original_text = text
    text = text.lower()
    
    # Remove HTML tags and XML-like structures
    text = re.sub(r'<[^>]*?>', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)  # HTML entities
    
    # Remove URLs and email addresses
    text = re.sub(r'http[s]?://[^\s<>"{}|\\^`\[\]]*', ' ', text)
    text = re.sub(r'www\.[^\s<>"{}|\\^`\[\]]*', ' ', text)
    text = re.sub(r'\S*@\S*\.\S*', ' ', text)
    
    # Remove phone numbers and dates
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' ', text)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', ' ', text)
    
    # Keep only A-Z, a-z, 0-9, and spaces as requested
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    
    # Remove standalone numbers and single letters
    text = re.sub(r'\b\d+\b(?!\s*(years?|months?|days?))', ' ', text)
    text = re.sub(r'\b[a-z]\b', ' ', text)  # Single letters
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Filter out very short or meaningless content
    if len(text.split()) < 15:  # Too short to be meaningful
        return ""
    
    return text

def create_jd_hash(text):
    """Create a hash for duplicate detection"""
    # Normalize text for comparison
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()

def extract_skills_with_patterns(jd_text):
    """Enhanced pattern-based skill extraction as fallback"""
    
    # Comprehensive skill patterns organized by category
    skill_patterns = {
        # Programming Languages
        "Python": r"\b(python|pandas|numpy|scipy|matplotlib|django|flask|fastapi)\b",
        "Java": r"\b(java|j2ee|spring|hibernate|maven|gradle)\b",
        "JavaScript": r"\b(javascript|js|node\.?js|react|angular|vue)\b",
        "C#": r"\b(c#|c-sharp|\.net|asp\.net)\b",
        "R": r"\br\b(?:\s+programming|\s+language|\s+statistical|,|$)",
        "Scala": r"\b(scala|spark)\b",
        "Go": r"\b(golang|go\s+programming)\b",
        
        # Databases
        "SQL": r"\b(sql|mysql|postgresql|postgres|oracle|sqlite|t-sql|pl/sql)\b",
        "MongoDB": r"\b(mongodb|mongo|nosql)\b",
        "Cassandra": r"\b(cassandra|apache cassandra)\b",
        "Redis": r"\b(redis|in-memory)\b",
        
        # Big Data & Analytics
        "Apache Spark": r"\b(spark|apache spark|pyspark)\b",
        "Hadoop": r"\b(hadoop|hdfs|mapreduce|hive|pig)\b",
        "Kafka": r"\b(kafka|apache kafka|streaming)\b",
        "Elasticsearch": r"\b(elasticsearch|elastic|kibana)\b",
        
        # Cloud Platforms
        "AWS": r"\b(aws|amazon web services|s3|ec2|lambda|redshift|emr|glue)\b",
        "Azure": r"\b(azure|microsoft azure|azure data factory|synapse)\b",
        "GCP": r"\b(gcp|google cloud|bigquery|dataflow)\b",
        
        # Data Integration Tools
        "Talend": r"\b(talend|talend open studio|talend data integration)\b",
        "Informatica": r"\b(informatica|powercenter|iics|idmc)\b",
        "SSIS": r"\b(ssis|sql server integration services)\b",
        "Apache NiFi": r"\b(nifi|apache nifi|data flow)\b",
        
        # Data Warehousing
        "Snowflake": r"\b(snowflake|snowsql|snowpipe)\b",
        "Redshift": r"\b(redshift|amazon redshift)\b",
        "BigQuery": r"\b(bigquery|google bigquery)\b",
        "Databricks": r"\b(databricks|delta lake)\b",
        
        # Visualization Tools
        "Tableau": r"\b(tableau|tableau desktop|tableau server)\b",
        "Power BI": r"\b(power\s?bi|powerbi|microsoft power bi)\b",
        "Looker": r"\b(looker|google looker)\b",
        "Qlik": r"\b(qlik|qlikview|qliksense)\b",
        
        # Concepts & Methodologies
        "ETL": r"\b(etl|elt|extract\s+transform\s+load|data\s+pipeline)\b",
        "Data Warehousing": r"\b(data\s+warehous\w*|dwh|edw|enterprise\s+data\s+warehouse)\b",
        "Data Modeling": r"\b(data\s+model\w*|dimensional\s+model\w*|star\s+schema|snowflake\s+schema)\b",
        "Machine Learning": r"\b(machine\s+learning|ml|artificial\s+intelligence|ai|deep\s+learning)\b",
        "Data Science": r"\b(data\s+scien\w*|predictive\s+analytics|statistical\s+analysis)\b",
        
        # DevOps & Tools
        "Docker": r"\b(docker|containerization|containers)\b",
        "Kubernetes": r"\b(kubernetes|k8s|container\s+orchestration)\b",
        "Git": r"\b(git|github|gitlab|version\s+control)\b",
        "Jenkins": r"\b(jenkins|ci/cd|continuous\s+integration)\b",
        
        # Methodologies
        "Agile": r"\b(agile|scrum|kanban|sprint)\b",
        "DevOps": r"\b(devops|continuous\s+delivery|automation)\b"
    }
    
    found_skills = {}
    
    for skill, pattern in skill_patterns.items():
        matches = re.findall(pattern, jd_text, re.IGNORECASE)
        if matches:
            # Count frequency but cap at reasonable level
            count = min(len(matches), 5)
            found_skills[skill] = count
    
    # Contextual inference for implicit skills
    if "data integration" in jd_text and "ETL" not in found_skills:
        found_skills["ETL"] = 1
    if "business intelligence" in jd_text and "Power BI" not in found_skills and "Tableau" not in found_skills:
        found_skills["Business Intelligence"] = 1
    if "cloud" in jd_text and not any(cloud in found_skills for cloud in ["AWS", "Azure", "GCP"]):
        found_skills["Cloud Computing"] = 1
    
    if not found_skills:
        return {
            "primary": "Data Analysis",  # Default fallback instead of empty
            "secondary": ["SQL", "Excel"],
            "confidence": "Low",
            "extraction_type": "pattern_fallback"
        }
    
    # Sort by frequency and relevance
    sorted_skills = sorted(found_skills.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_skills[0][0]
    secondary = [skill for skill, _ in sorted_skills[1:6]]  # Top 5 secondary
    
    return {
        "primary": primary,
        "secondary": secondary,
        "confidence": "Medium",
        "extraction_type": "pattern_based"
    }

def predict_skills_with_claude(jd_text):
    """Enhanced skill prediction using Claude API with strict validation"""
    if not claude_headers:
        logger.warning("Claude API not available, falling back to pattern matching")
        return extract_skills_with_patterns(jd_text)
    
    try:
        # Prepare the Claude API request
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 500,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": f"""Given the job description below, extract the Primary Skill and 5+ Secondary Skills. 

Primary Skill is the one the role critically depends on (exactly ONE skill).
Secondary Skills are supporting or nice-to-have skills (at least 5 skills).

STRICT RULES:
- Extract ONLY explicitly mentioned technical skills, tools, or technologies
- NEVER use placeholders like "None specified", "Not mentioned", "TBD", etc.
- For ambiguous descriptions, intelligently infer from context
- Prioritize concrete tools/technologies over soft skills
- Use standard industry names (e.g., "AWS" not "Amazon Web Services")
- If no technical skills are clearly mentioned, infer the most likely skills based on job context

JOB DESCRIPTION:
{jd_text}

Return in this exact format:
Primary Skill: <single most critical technical skill>
Secondary Skills: <skill1>, <skill2>, <skill3>, <skill4>, <skill5>, <additional skills if any>

Example:
Primary Skill: Python
Secondary Skills: SQL, AWS, Docker, Machine Learning, Git, Pandas
"""
                }
            ]
        }
        
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=claude_headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"Claude API error: {response.status_code} - {response.text}")
            return extract_skills_with_patterns(jd_text)
        
        result = response.json()
        content = result['content'][0]['text'].strip()
        
        # Parse Claude's response
        primary_skill = ""
        secondary_skills = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith('primary skill:'):
                primary_skill = line.split(':', 1)[1].strip()
            elif line.lower().startswith('secondary skills:'):
                skills_text = line.split(':', 1)[1].strip()
                secondary_skills = [s.strip() for s in skills_text.split(',') if s.strip()]
        
        # Validate and clean the output
        forbidden_values = ["none", "not specified", "not mentioned", "tbd", "n/a", "na", ""]
        
        if primary_skill.lower() in forbidden_values:
            primary_skill = ""
        
        secondary_skills = [s for s in secondary_skills if s.lower() not in forbidden_values]
        
        # If Claude returns insufficient results, supplement with pattern matching
        if not primary_skill or len(secondary_skills) < 2:
            pattern_result = extract_skills_with_patterns(jd_text)
            if not primary_skill:
                primary_skill = pattern_result["primary"]
            if len(secondary_skills) < 2:
                secondary_skills.extend(pattern_result["secondary"])
                secondary_skills = list(dict.fromkeys(secondary_skills))[:7]  # Remove duplicates, keep top 7
        
        return {
            "primary": primary_skill if primary_skill else "Data Analysis",
            "secondary": secondary_skills if secondary_skills else ["SQL", "Excel"],
            "confidence": "High" if primary_skill and len(secondary_skills) >= 5 else "Medium",
            "extraction_type": "claude_api",
            "raw_response": content
        }

    except requests.RequestException as e:
        logger.error(f"Claude API request error: {e}")
        return extract_skills_with_patterns(jd_text)
    except Exception as e:
        logger.error(f"Claude API processing error: {e}")
        return extract_skills_with_patterns(jd_text)

@csrf_exempt
def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        file_path = None
        original_count = 0
        
        try:
            # Save and read the uploaded file
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)
            
            # Read file based on extension and store original count
            if filename.lower().endswith('.csv'):
                df_original = pd.read_csv(file_path, encoding='utf-8')
            else:
                df_original = pd.read_excel(file_path)
            
            original_count = len(df_original)
            logger.info(f"Loaded file with {original_count} rows and columns: {df_original.columns.tolist()}")

            # Find JD column with more flexible matching
            jd_column = None
            for col in df_original.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in [
                    'job', 'description', 'jd', 'role', 'position', 
                    'requirement', 'responsibility', 'duty', 'summary'
                ]):
                    jd_column = col
                    break
            
            if not jd_column:
                # If no obvious column, use the first text column with substantial content
                for col in df_original.columns:
                    if df_original[col].dtype == 'object':  # Text column
                        avg_length = df_original[col].astype(str).str.len().mean()
                        if avg_length > 100:  # Assume JD if average length > 100 chars
                            jd_column = col
                            break
            
            if not jd_column:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                return JsonResponse({
                    "status": "error",
                    "message": f"No job description column found. Available columns: {', '.join(df_original.columns.tolist())}"
                }, status=400)

            logger.info(f"Using column '{jd_column}' as job description")

            # Step 1: Drop NaN entries
            df = df_original[[jd_column]].copy()
            df = df.dropna()  # Remove null values
            
            # Step 2: Advanced cleaning (keep only A-Z, a-z, 0-9, spaces)
            df['cleaned'] = df[jd_column].apply(advanced_clean_jd_text)
            
            # Remove empty cleaned descriptions
            df = df[df['cleaned'].str.len() > 0]
            
            # Step 3: Remove duplicates based on content hash
            df['hash'] = df['cleaned'].apply(create_jd_hash)
            df = df.drop_duplicates(subset=['hash'])
            
            # Filter out very short descriptions
            df = df[df['cleaned'].str.split().str.len() >= 20]  # At least 20 words
            
            logger.info(f"After preprocessing: {len(df)} unique job descriptions remain")

            if len(df) == 0:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                return JsonResponse({
                    "status": "error",
                    "message": "No valid job descriptions found after preprocessing"
                }, status=400)

            # Step 4: Process each JD with Claude LLM skill extraction
            results = []
            for idx, row in df.iterrows():
                skills = predict_skills_with_claude(row['cleaned'])
                
                # Create result entry
                result_entry = {
                    "id": idx,
                    "job_description": row[jd_column][:500] + ("..." if len(row[jd_column]) > 500 else ""),
                    "primary_skill": skills["primary"],
                    "secondary_skills": skills["secondary"],
                    "confidence": skills["confidence"],
                    "source": skills["extraction_type"],
                    "word_count": len(row['cleaned'].split())
                }
                
                # Add raw response if available (for debugging)
                if "raw_response" in skills:
                    result_entry["claude_response"] = skills["raw_response"]
                
                results.append(result_entry)

            # Generate summary statistics
            all_primary = [r["primary_skill"] for r in results if r["primary_skill"]]
            all_secondary = []
            for r in results:
                all_secondary.extend(r["secondary_skills"])
            
            primary_counter = Counter(all_primary)
            secondary_counter = Counter(all_secondary)

            # Clean up file
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            return JsonResponse({
                "status": "success",
                "results": results,
                "summary": {
                    "total_processed": len(results),
                    "file_name": uploaded_file.name,
                    "top_primary_skills": dict(primary_counter.most_common(10)),
                    "top_secondary_skills": dict(secondary_counter.most_common(15)),
                    "extraction_methods": Counter([r["source"] for r in results]),
                    "preprocessing_stats": {
                        "original_count": original_count,
                        "after_cleaning": len(results)
                    }
                }
            })

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            # Clean up file if it exists
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as cleanup_error:
                    logger.error(f"Error cleaning up file: {cleanup_error}")
            
            return JsonResponse({
                "status": "error",
                "message": f"Processing failed: {str(e)}"
            }, status=500)

    return JsonResponse({
        "status": "error",
        "message": "Invalid request. Please upload a CSV or Excel file."
    }, status=400)

@csrf_exempt
def analyze_single_jd(request):
    """Endpoint for analyzing a single job description via text input"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            jd_text = data.get('job_description', '')
            
            if not jd_text or len(jd_text.strip()) < 50:
                return JsonResponse({
                    "status": "error", 
                    "message": "Please provide a job description with at least 50 characters"
                }, status=400)
            
            cleaned_text = advanced_clean_jd_text(jd_text)
            skills = predict_skills_with_claude(cleaned_text)
            
            return JsonResponse({
                "status": "success",
                "result": {
                    "primary_skill": skills["primary"],
                    "secondary_skills": skills["secondary"],
                    "confidence": skills["confidence"],
                    "source": skills["extraction_type"],
                    "raw_response": skills.get("raw_response", ""),
                    "word_count": len(cleaned_text.split())
                }
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                "status": "error",
                "message": "Invalid JSON format"
            }, status=400)
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)
    
    return JsonResponse({
        "status": "error",
        "message": "Only POST method allowed"
    }, status=405)

@csrf_exempt  
def debug_claude(request):
    """Diagnostic endpoint for Claude API"""
    status = {
        'api_key_configured': bool(hasattr(settings, 'CLAUDE_API_KEY') and settings.CLAUDE_API_KEY),
        'api_key_prefix': settings.CLAUDE_API_KEY[:8] + '...' if hasattr(settings, 'CLAUDE_API_KEY') and settings.CLAUDE_API_KEY else None,
        'client_initialized': bool(claude_headers)
    }
    
    if claude_headers:
        try:
            test_payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}]
            }
            
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=claude_headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                status['connection_test'] = "Success"
            else:
                status['connection_error'] = f"HTTP {response.status_code}: {response.text}"
                
        except Exception as e:
            status['connection_error'] = str(e)
    
    return JsonResponse(status)