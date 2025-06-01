from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.conf import settings
import openai
import pandas as pd
import json
import os

openai.api_key = settings.OPENAI_API_KEY

def index(request):
    return render(request, 'index.html')

def extract_skills_from_jd(jd_text):
    prompt = f"""
You are an expert IT recruiter and HR analyst.

Analyze the following job description and extract the **Primary Skill** and **Secondary Skills** that clearly match the JD.

Job Description:
\"\"\"{jd_text}\"\"\"

Respond strictly in this JSON format:
{{
  "Primary Skill": "Your one most important skill",
  "Secondary Skills": ["Skill1", "Skill2", "Skill3"]
}}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract skills from job descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        content = response['choices'][0]['message']['content']
        result = json.loads(content)
        return {
            "primary": result.get("Primary Skill", "N/A"),
            "secondary": result.get("Secondary Skills", [])
        }
    except Exception as e:
        return {
            "primary": "Error",
            "secondary": [f"Error: {str(e)}"]
        }

def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        try:
            df = pd.read_excel(file_path)

            jd_column = None
            for col in df.columns:
                if col.strip().lower() in ["job description", "jd"]:
                    jd_column = col
                    break

            if not jd_column:
                os.remove(file_path)
                return JsonResponse({"status": "error", "message": "No 'Job description' column found in the uploaded file."})

            df = df[[jd_column]].dropna()
            df[jd_column] = df[jd_column].astype(str).str.strip()
            df.drop_duplicates(subset=[jd_column], inplace=True)
            df = df[df[jd_column] != ""]  # Remove empty strings

            results = []
            for jd in df[jd_column]:
                if len(jd.split()) < 5:
                    continue  # Skip short/incomplete descriptions
                skills = extract_skills_from_jd(jd)
                results.append({
                    "jd": jd,
                    "primary": skills["primary"],
                    "secondary": skills["secondary"]
                })

            os.remove(file_path)
            return JsonResponse({"status": "success", "results": results})

        except Exception as e:
            os.remove(file_path)
            return JsonResponse({"status": "error", "message": str(e)})

    return JsonResponse({"status": "error", "message": "Invalid request"})
