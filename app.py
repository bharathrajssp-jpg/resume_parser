import streamlit as st
import re
import json
import pandas as pd
from datetime import datetime
import io

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("PyMuPDF not installed. PDF support disabled.")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("python-docx not installed. DOCX support disabled.")

class ResumeParser:
    def __init__(self):
        self.skills_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'html', 'css', 'react', 
            'angular', 'vue', 'node.js', 'nodejs', 'django', 'flask', 'spring', 'aws', 
            'azure', 'gcp', 'docker', 'kubernetes', 'git', 'github', 'gitlab', 
            'machine learning', 'deep learning', 'data analysis', 'excel', 'powerbi', 
            'tableau', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 
            'mongodb', 'postgresql', 'mysql', 'redis', 'api', 'rest', 'graphql', 
            'agile', 'scrum', 'jira', 'ci/cd', 'linux', 'unix', 'bash', 'typescript', 
            'go', 'rust', 'kotlin', 'swift', 'spark', 'hadoop', 'opencv', 'selenium'
        ]
        
        self.degree_keywords = [
            'B.Tech', 'M.Tech', 'B.E.', 'M.E.', 'B.Sc', 'M.Sc', 'MBA', 'BBA', 
            'Bachelor', 'Master', 'PhD', 'Ph.D', 'B.A.', 'M.A.', 'B.Com', 'M.Com'
        ]

    def extract_text_from_pdf(self, file):
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            st.error("PDF support not available. Install PyMuPDF: pip install pymupdf")
            return ""
        
        try:
            file.seek(0)
            file_bytes = file.read()
            
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
                text = ""
                for page in pdf_doc:
                    text += page.get_text()
            
            return text if text.strip() else ""
            
        except Exception as e:
            st.error(f"PDF extraction error: {str(e)}")
            return ""

    def extract_text_from_docx(self, file):
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            st.error("DOCX support not available. Install python-docx: pip install python-docx")
            return ""
        
        try:
            file.seek(0)
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text if text.strip() else ""
            
        except Exception as e:
            st.error(f"DOCX extraction error: {str(e)}")
            return ""

    def extract_email(self, text):
        """Extract email address"""
        try:
            pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(pattern, text)
            return emails[0] if emails else "Not found"
        except:
            return "Not found"

    def extract_phone(self, text):
        """Extract phone number"""
        try:
            patterns = [
                r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
                r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                r'\d{10}'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    clean = re.sub(r'[^\d+]', '', match)
                    if len(clean) >= 10:
                        return match.strip()
            
            return "Not found"
        except:
            return "Not found"

    def extract_name(self, text):
        """Extract candidate name from resume"""
        try:
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            
            if not lines:
                return "Not found"
            
            skip_terms = ['resume', 'cv', 'curriculum', 'email', 'phone', 'address', 
                         'linkedin', 'github', 'portfolio', 'objective', 'summary', 
                         'profile', 'http', 'www', '@']
            
            for line in lines[:10]:
                # Skip unwanted lines
                if any(term in line.lower() for term in skip_terms):
                    continue
                
                if '@' in line or line[0].isdigit() if line else False:
                    continue
                
                # Check line quality
                if not (3 <= len(line) <= 50):
                    continue
                
                # Count character types
                letters = sum(c.isalpha() or c.isspace() for c in line)
                digits = sum(c.isdigit() for c in line)
                
                if digits > 3:
                    continue
                
                # Should be mostly letters
                if letters / len(line) >= 0.7:
                    words = line.split()
                    # Name should have 1-4 words
                    if 1 <= len(words) <= 4:
                        clean_name = ' '.join(words)
                        if len(clean_name) >= 3:
                            return clean_name
            
            return "Not found"
        except:
            return "Not found"

    def extract_skills(self, text):
        """Extract technical skills"""
        try:
            text_lower = text.lower()
            found = set()
            
            for skill in self.skills_keywords:
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found.add(skill.title())
            
            return sorted(list(found))
        except:
            return []

    def get_sections(self, text):
        """Identify section boundaries in resume"""
        lines = text.split('\n')
        sections = {}
        
        section_map = {
            'education': ['education', 'academic background', 'qualifications'],
            'experience': ['experience', 'work experience', 'employment', 'work history'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'projects': ['projects'],
            'certifications': ['certifications', 'certificates']
        }
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            
            if not line_clean or len(line_clean) > 40:
                continue
            
            for section, keywords in section_map.items():
                if any(kw == line_clean or line_clean.startswith(kw) for kw in keywords):
                    if line.count('.') <= 1 and line.count(',') == 0:
                        sections[section] = i
                        break
        
        return sections, lines

    def extract_education(self, text):
        """Extract education details"""
        try:
            sections, lines = self.get_sections(text)
            education_list = []
            
            if 'education' in sections:
                start = sections['education']
                end = min([idx for idx in sections.values() if idx > start] + [len(lines)])
                content = '\n'.join(lines[start+1:end])
            else:
                content = text
            
            # Pattern 1: Full education entry
            pattern1 = r'(B\.?Tech|M\.?Tech|B\.?E\.?|M\.?E\.?|B\.?Sc|M\.?Sc|MBA|BBA|Bachelor|Master|PhD).*?(?:University|College|Institute).*?(\d{4})'
            
            matches = re.finditer(pattern1, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                entry = match.group(0).strip()
                # Clean up the entry
                entry = ' '.join(entry.split())
                if 25 < len(entry) < 200:
                    education_list.append(entry)
            
            # Pattern 2: Degree with year
            if len(education_list) < 2:
                pattern2 = r'(B\.?Tech|M\.?Tech|B\.?Sc|M\.?Sc|MBA|Bachelor|Master).*?(\d{4}\s*-\s*\d{4}|\d{4})'
                matches = re.finditer(pattern2, content, re.IGNORECASE)
                
                for match in matches:
                    text_snippet = match.group(0).strip()
                    # Get surrounding context
                    start_pos = match.start()
                    end_pos = min(match.end() + 100, len(content))
                    context = content[start_pos:end_pos]
                    
                    # Extract university if present
                    uni_match = re.search(r'[A-Z][A-Za-z\s,&-]+(?:University|College|Institute)', context)
                    if uni_match:
                        full_entry = f"{match.group(1)} - {uni_match.group(0)} ({match.group(2)})"
                        full_entry = ' '.join(full_entry.split())
                        if full_entry not in education_list and len(full_entry) > 20:
                            education_list.append(full_entry)
            
            # Remove duplicates
            unique_edu = []
            for item in education_list:
                is_dup = any(item.lower() in existing.lower() or 
                           existing.lower() in item.lower() 
                           for existing in unique_edu)
                if not is_dup:
                    unique_edu.append(item)
            
            return unique_edu[:5] if unique_edu else ["Not found"]
            
        except Exception as e:
            st.warning(f"Education extraction issue: {str(e)}")
            return ["Not found"]

    def extract_experience(self, text):
        """Extract work experience"""
        try:
            sections, lines = self.get_sections(text)
            experience_list = []
            
            if 'experience' not in sections:
                return ["Not found"]
            
            start = sections['experience']
            end = min([idx for idx in sections.values() if idx > start] + [len(lines)])
            section_lines = lines[start+1:end]
            
            current_job = None
            
            for line in section_lines[:25]:
                line = line.strip()
                if not line:
                    continue
                
                # Check for date patterns (indicates job entry)
                date_pattern = r'(\d{4})\s*[-–—]\s*(\d{4}|Present|Current|Now)'
                date_match = re.search(date_pattern, line, re.IGNORECASE)
                
                if date_match:
                    # Save previous job
                    if current_job and len(current_job) > 20:
                        experience_list.append(current_job)
                    
                    # Start new job entry
                    current_job = line
                elif current_job:
                    # Add description (limit length)
                    if len(current_job) < 150:
                        current_job += " | " + line
            
            # Add last job
            if current_job and len(current_job) > 20:
                experience_list.append(current_job)
            
            # Clean up entries
            cleaned = []
            for exp in experience_list:
                exp = ' '.join(exp.split())
                if len(exp) > 20:
                    cleaned.append(exp)
            
            return cleaned[:6] if cleaned else ["Not found"]
            
        except Exception as e:
            st.warning(f"Experience extraction issue: {str(e)}")
            return ["Not found"]

    def parse_resume(self, file, file_type):
        """Main parsing function"""
        try:
            # Extract text based on file type
            if file_type == 'pdf':
                text = self.extract_text_from_pdf(file)
            elif file_type == 'docx':
                text = self.extract_text_from_docx(file)
            else:
                st.error(f"Unsupported file type: {file_type}")
                return None
            
            if not text:
                st.error("Could not extract text from file")
                return None
            
            # Parse all fields
            result = {
                'name': self.extract_name(text),
                'email': self.extract_email(text),
                'phone': self.extract_phone(text),
                'skills': self.extract_skills(text),
                'education': self.extract_education(text),
                'experience': self.extract_experience(text),
                'raw_text_preview': text[:500]
            }
            
            return result
            
        except Exception as e:
            st.error(f"Parsing error: {str(e)}")
            return None


def main():
    st.set_page_config(
        page_title="Resume Parser",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Resume Parser Pro")
    st.markdown("### AI-powered resume information extraction tool")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **Extracts:**
        - 👤 Name & Contact
        - 🎯 Skills
        - 🎓 Education
        - 💼 Experience
        
        **Formats:** PDF, DOCX
        """)
        
        st.header("📝 Usage")
        st.markdown("""
        1. Upload resume(s)
        2. View extracted data
        3. Download results
        """)
    
    # Initialize parser
    parser = ResumeParser()
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Resume Files",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="Select PDF or DOCX files"
    )
    
    if not uploaded_files:
        st.info("👆 Upload resume files to begin parsing")
        return
    
    # Process files
    all_results = []
    
    for file in uploaded_files:
        st.markdown(f"### 📄 {file.name}")
        
        # Determine file type
        file_ext = file.name.lower().split('.')[-1]
        
        with st.spinner(f"Processing {file.name}..."):
            result = parser.parse_resume(file, file_ext)
        
        if not result:
            st.error(f"Failed to parse {file.name}")
            continue
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👤 Personal Info**")
            st.text(f"Name: {result['name']}")
            st.text(f"Email: {result['email']}")
            st.text(f"Phone: {result['phone']}")
        
        with col2:
            st.markdown("**🎯 Skills**")
            if result['skills']:
                st.write(' '.join([f"`{s}`" for s in result['skills'][:15]]))
                if len(result['skills']) > 15:
                    st.caption(f"+ {len(result['skills']) - 15} more")
            else:
                st.text("No skills detected")
        
        st.markdown("**🎓 Education**")
        if result['education'] and result['education'] != ["Not found"]:
            for i, edu in enumerate(result['education'], 1):
                st.text(f"{i}. {edu}")
        else:
            st.text("No education found")
        
        st.markdown("**💼 Experience**")
        if result['experience'] and result['experience'] != ["Not found"]:
            for i, exp in enumerate(result['experience'], 1):
                st.text(f"{i}. {exp[:150]}...")
        else:
            st.text("No experience found")
        
        with st.expander("🔍 Raw Text"):
            st.text(result['raw_text_preview'])
        
        result['filename'] = file.name
        all_results.append(result)
        st.divider()
    
    # Export section
    if all_results:
        st.markdown("---")
        st.header("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        # JSON export
        with col1:
            export_json = []
            for r in all_results:
                export_json.append({
                    'filename': r['filename'],
                    'name': r['name'],
                    'email': r['email'],
                    'phone': r['phone'],
                    'skills': r['skills'],
                    'education': r['education'],
                    'experience': r['experience']
                })
            
            json_data = json.dumps(export_json, indent=2)
            st.download_button(
                "📄 Download JSON",
                json_data,
                f"resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
        
        # CSV export
        with col2:
            csv_rows = []
            for r in all_results:
                csv_rows.append({
                    'File': r['filename'],
                    'Name': r['name'],
                    'Email': r['email'],
                    'Phone': r['phone'],
                    'Skills': ', '.join(r['skills']) if r['skills'] else '',
                    'Education': ' | '.join(r['education']) if r['education'] else '',
                    'Experience': ' | '.join(r['experience']) if r['experience'] else ''
                })
            
            df = pd.DataFrame(csv_rows)
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                "📊 Download CSV",
                csv_data,
                f"resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        
        # Summary stats
        st.markdown("---")
        st.subheader("📊 Summary")
        
        stat1, stat2, stat3, stat4 = st.columns(4)
        
        with stat1:
            st.metric("Files Processed", len(all_results))
        
        with stat2:
            names = sum(1 for r in all_results if r['name'] != "Not found")
            st.metric("Names Found", names)
        
        with stat3:
            emails = sum(1 for r in all_results if r['email'] != "Not found")
            st.metric("Emails Found", emails)
        
        with stat4:
            avg_skills = sum(len(r['skills']) for r in all_results) / len(all_results)
            st.metric("Avg Skills", f"{avg_skills:.1f}")
        
        # Display summary table
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()