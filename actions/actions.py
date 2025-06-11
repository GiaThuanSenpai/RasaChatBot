from typing import Any, Text, Dict, List
import pandas as pd
import os
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from fuzzywuzzy import fuzz
from rasa_sdk.events import SlotSet
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import unicodedata
import logging

logging.basicConfig(level=logging.DEBUG)

class ActionSearchJobs(Action):

    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            stop_words=['và', 'hoặc', 'của', 'với', 'các', 'những']
        )
        self.corpus = self._initialize_corpus()
        self.vectors = None
        self._update_vectors()
        self.INDUSTRY_KEYWORDS = {
            'thương mại điện tử': ['thương mại điện tử', 'bán hàng trực tuyến', 'e-commerce', 'sàn thương mại', 'mua sắm online', 'tmdt', 'digital commerce', 'marketplace'],
            'marketing': ['marketing', 'truyền thông', 'quảng cáo', 'tiếp thị', 'thương hiệu', 'pr', 'digital marketing', 'seo', 'sem', 'content marketing'],
            'it phần cứng': ['it phần cứng', 'kỹ thuật phần cứng', 'công nghệ phần cứng', 'hardware', 'network engineering', 'mạng máy tính', 'hạ tầng mạng', 'server', 'network infrastructure'],
            'công nghệ ô tô': ['công nghệ ô tô', 'kỹ thuật ô tô', 'sửa chữa ô tô', 'automotive technology', 'xe hơi', 'bảo dưỡng xe', 'car technology'],
            'it phần mềm': ['it phần mềm', 'phần mềm', 'software', 'lập trình', 'phát triển phần mềm', 'công nghệ thông tin', 'software development', 'cntt', 'developer', 'programmer', 'coding'],
            'nhà hàng': ['nhà hàng', 'khách sạn', 'dịch vụ ăn uống', 'hospitality', 'f&b', 'quản lý nhà hàng', 'đầu bếp', 'restaurant'],
            'thiết kế': ["thiết kế", "in ấn", "đồ họa", "graphic design", "ui/ux", "thiết kế giao diện", "thiết kế web", "design", "creative"],
            'cơ khí': ['cơ khí', 'điện - điện tử', 'mechanical engineering', 'electrical engineering', 'điện tử', 'tự động hóa', 'automation'],
            'kinh doanh': ['kinh doanh', 'bán hàng', 'thương mại', 'business', 'sales', 'phát triển kinh doanh', 'xuất nhập khẩu', 'commerce'],
            'giáo dục': ['giáo dục', 'đào tạo', 'giảng dạy', 'education', 'teaching', 'giáo viên', 'đào tạo doanh nghiệp', 'training'],
            'kiến trúc': ['kiến trúc', 'xây dựng', 'công trình', 'architecture', 'construction', 'kỹ sư xây dựng', 'civil engineering'],
            'tài chính': ['tài chính', 'ngân hàng', 'đầu tư', 'finance', 'banking', 'chứng khoán', 'phân tích tài chính', 'investment'],
            'viễn thông': ['viễn thông', 'mạng viễn thông', 'telecommunications', 'telecom', 'mạng di động', 'network telecom'],
            'y tế': ['y tế', 'chăm sóc sức khỏe', 'điều dưỡng', 'healthcare', 'medical', 'bác sĩ', 'dược sĩ', 'hospital'],
            'logistics': ['logistics', 'vận tải', 'chuỗi cung ứng', 'supply chain', 'warehousing', 'quản lý kho', 'vận chuyển', 'shipping'],
            'kế toán': ['kế toán', 'kiểm toán', 'báo cáo tài chính', 'accounting', 'auditing', 'thuế', 'bookkeeping', 'finance'],
            'sản xuất': ['sản xuất', 'chế tạo', 'công nghiệp', 'manufacturing', 'production', 'quản lý sản xuất', 'factory'],
            'tài xế': ['tài xế', 'lái xe', 'vận chuyển', 'driver', 'giao hàng', 'shipper', 'delivery'],
            'luật': ['luật', 'pháp lý', 'tư vấn pháp luật', 'law', 'legal', 'luật sư', 'pháp chế', 'legal counsel'],
            'phiên dịch': ['phiên dịch', 'dịch thuật', 'thông dịch', 'translation', 'interpretation', 'ngôn ngữ', 'translator'],
            'hệ thống nhúng': ['hệ thống nhúng', 'embedded systems', 'iot', 'firmware', 'microcontroller', 'RTOS', 'ARM', 'FPGA', 'sensor programming', 'device driver', 'low-level programming', 'embedded software']
        }
        self.SPECIALIZATION_KEYWORDS = {
            'lập trình': ['developer', 'programmer', 'software engineer', 'frontend', 'backend', 'fullstack', 'devops', 'lập trình viên', 'kỹ sư phần mềm', 'phát triển phần mềm', 'software development', 'coding'],
            'quảng cáo': ['quảng cáo', 'content marketing', 'copywriting', 'social media', 'seo', 'sem', 'digital marketing', 'tiếp thị nội dung'],
            'thiết kế': ['thiết kế', 'graphic designer', 'ui/ux', 'illustrator', 'photoshop', 'thiết kế đồ họa', 'thiết kế giao diện', 'ux designer', 'design'],
            'kỹ thuật': ['kỹ thuật', 'engineer', 'technician', 'maintenance', 'kỹ sư', 'bảo trì', 'kỹ thuật viên', 'engineering'],
            'bán hàng': ['bán hàng', 'sales', 'business development', 'crm', 'đại diện bán hàng', 'phát triển kinh doanh', 'sales representative'],
            'giáo viên': ['giáo viên', 'giảng viên', 'teacher', 'instructor', 'tutor', 'đào tạo', 'education'],
            'kiến trúc sư': ['kiến trúc sư', 'architect', 'civil engineer', 'cad', 'kỹ sư xây dựng', 'architecture'],
            'tài chính': ['phân tích tài chính', 'financial analyst', 'investment', 'credit', 'quản lý tài chính', 'tư vấn đầu tư', 'finance'],
            'điều dưỡng': ['điều dưỡng', 'y tá', 'nurse', 'healthcare', 'chăm sóc sức khỏe', 'medical'],
            'tài xế': ['tài xế', 'lái xe', 'driver', 'vận chuyển', 'giao hàng', 'delivery'],
            'luật sư': ['luật sư', 'pháp chế', 'legal counsel', 'tư vấn pháp luật', 'cố vấn pháp lý', 'law'],
            'phiên dịch': ['phiên dịch', 'biên dịch', 'translator', 'interpreter', 'dịch thuật', 'translation'],
            'hệ thống nhúng': ['embedded engineer', 'firmware engineer', 'iot developer', 'microcontroller programming', 'embedded software', 'real-time systems', 'embedded systems'],
            'phân tích dữ liệu': ['data analyst', 'phân tích dữ liệu', 'business intelligence', 'data scientist', 'sql', 'python', 'data analysis'],
            'quản lý dự án': ['project manager', 'quản lý dự án', 'pm', 'scrum master', 'agile', 'project management'],
            'kiểm thử phần mềm': ['qa', 'quality assurance', 'tester', 'kiểm thử phần mềm', 'test engineer', 'software testing'],
            'hỗ trợ khách hàng': ['hỗ trợ khách hàng', 'customer support', 'customer service', 'support'],
            'an ninh mạng': ['cybersecurity', 'an ninh mạng', 'network security', 'ethical hacker', 'information security'],
            'nhân sự': ['human resources', 'nhân sự', 'hr', 'recruitment', 'tuyển dụng', 'talent acquisition'],
            'logistics': ['logistics', 'vận hành kho', 'warehouse', 'supply chain', 'quản lý vận tải', 'shipping'],
            'chăm sóc sức khỏe': ['healthcare', 'chăm sóc sức khỏe', 'bác sĩ', 'doctor', 'y sĩ', 'medical'],
            'kế toán': ['kế toán', 'accountant', 'bookkeeper', 'báo cáo tài chính', 'financial reporting', 'accounting']
        }
        self.NEGATIVE_KEYWORDS = {
            'hệ thống nhúng': ['network', 'mạng máy tính', 'network engineering', 'hệ thống mạng', 'network infrastructure', 'server', 'cloud computing'],
            'it phần mềm': ['hardware', 'mạng máy tính', 'network engineering'],
            'it phần cứng': ['software', 'phát triển phần mềm', 'lập trình'],
            'marketing': ['sản xuất', 'manufacturing'],
            'thương mại điện tử': ['sản xuất', 'cơ khí'],
            'kinh doanh': ['kỹ thuật', 'lập trình'],
            'giáo dục': ['sản xuất', 'vận tải'],
            'tài chính': ['cơ khí', 'sản xuất'],
            'y tế': ['kinh doanh', 'bán hàng'],
            'logistics': ['lập trình', 'phát triển phần mềm'],
            'kế toán': ['quảng cáo', 'marketing'],
            'luật': ['cơ khí', 'sản xuất'],
            'phiên dịch': ['lập trình', 'kỹ thuật']
        }

    def name(self) -> Text:
        return "action_search_jobs"

    def _initialize_corpus(self):
        return {
            'thương mại điện tử': ['thương mại điện tử ecommerce tmdt online shopping marketplace sàn bán hàng online digital commerce'],
            'marketing': ['marketing truyền thông quảng cáo pr digital marketing social media content marketing seo sem'],
            'it phần cứng': ['it phần cứng hardware network engineering mạng máy tính server network infrastructure'],
            'công nghệ ô tô': ['công nghệ ô tô kỹ thuật ô tô automotive technology xe hơi sửa chữa ô tô bảo dưỡng xe'],
            'it phần mềm': ['it phần mềm công nghệ thông tin lập trình viên developer programmer software engineer frontend backend fullstack devops mobile app web development coding cntt phát triển phần mềm software development'],
            'nhà hàng': ['nhà hàng khách sạn hospitality restaurant food beverage f&b dịch vụ ăn uống quản lý nhà hàng chef'],
            'thiết kế': ['thiết kế in ấn graphic design ui ux designer printing nghệ thuật đồ họa thiết kế đồ họa thiết kế web'],
            'cơ khí': ['cơ khí điện điện tử mechanical engineering electrical engineering kỹ sư cơ khí tự động hóa'],
            'kinh doanh': ['kinh doanh bán hàng sales business development thương mại retail xuất nhập khẩu'],
            'giáo dục': ['giáo dục đào tạo education teaching teacher giảng viên giáo viên training'],
            'kiến trúc': ['kiến trúc xây dựng architecture construction civil engineering thiết kế kiến trúc'],
            'tài chính': ['tài chính ngân hàng finance banking investment chứng khoán financial analyst đầu tư'],
            'viễn thông': ['viễn thông telecommunications telecom mạng viễn thông network infrastructure'],
            'y tế': ['y tế healthcare medical bác sĩ doctor nurse điều dưỡng bệnh viện hospital'],
            'logistics': ['logistics chuỗi cung ứng supply chain warehouse quản lý kho vận tải shipping'],
            'kế toán': ['kế toán kiểm toán accounting auditing financial reporting báo cáo tài chính thuế'],
            'sản xuất': ['sản xuất manufacturing production nhà máy quản lý sản xuất quality control'],
            'tài xế': ['tài xế lái xe driver vận chuyển delivery giao hàng shipper'],
            'luật': ['luật pháp lý legal law lawyer tư vấn pháp luật luật sư'],
            'phiên dịch': ['phiên dịch biên dịch translation interpreter ngôn ngữ dịch thuật'],
            'hệ thống nhúng': ['hệ thống nhúng embedded systems iot firmware microcontroller RTOS ARM FPGA sensor programming device driver embedded software']
        }

    def _add_synonyms(self, text: str) -> str:
        synonyms = {
            'công nghệ thông tin': 'it phần mềm lập trình developer programmer coding software cntt kỹ sư phần mềm software development',
            'lập trình': 'developer programmer coding software engineering code kỹ sư phần mềm software development',
            'kinh doanh': 'sales bán hàng thương mại business phát triển kinh doanh commerce',
            'nhân sự': 'human resources hr tuyển dụng recruitment quản lý nhân sự',
            'kế toán': 'accounting accountant finance tài chính kiểm toán báo cáo tài chính',
            'tiếp thị': 'marketing quảng cáo advertising truyền thông pr digital marketing',
            'thiết kế': 'design graphic ui ux đồ họa mỹ thuật sáng tạo giao diện',
            'quản lý': 'manager management leader quản trị điều hành giám đốc',
            'giáo dục': 'education teaching giảng dạy đào tạo giáo viên giảng viên',
            'y tế': 'healthcare medical bác sĩ nurse điều dưỡng chăm sóc sức khỏe',
            'xây dựng': 'construction kiến trúc công trình kỹ sư xây dựng thi công',
            'logistics': 'chuỗi cung ứng supply chain vận tải kho bãi hậu cần',
            'luật': 'pháp lý legal luật sư tư vấn pháp luật cố vấn pháp lý',
            'phiên dịch': 'dịch thuật translation thông dịch interpreter ngôn ngữ',
            'tài chính': 'finance banking investment ngân hàng chứng khoán phân tích tài chính',
            'hệ thống nhúng': 'embedded systems iot firmware microcontroller RTOS ARM FPGA sensor programming device driver',
            'thương mại điện tử': 'e-commerce tmdt mua sắm online sàn thương mại digital commerce',
            'marketing': 'tiếp thị quảng cáo truyền thông pr digital marketing seo sem',
            'cơ khí': 'mechanical engineering điện điện tử tự động hóa kỹ thuật cơ khí',
            'viễn thông': 'telecommunications telecom mạng viễn thông network telecom',
            'tài xế': 'lái xe driver vận chuyển giao hàng shipper'
        }
        words = text.lower().split()
        expanded_words = words.copy()
        for word in words:
            if word in synonyms:
                expanded_words.extend(synonyms[word].split())
        return ' '.join(set(expanded_words))

    def _expand_abbreviations(self, text: str) -> str:
        abbreviations = {
            'cntt': 'công nghệ thông tin',
            'it': 'công nghệ thông tin',
            'hr': 'nhân sự',
            'kd': 'kinh doanh',
            'kt': 'kế toán',
            'marketing': 'tiếp thị',
            'dev': 'developer lập trình',
            'qa': 'kiểm thử phần mềm',
            'ba': 'phân tích nghiệp vụ',
            'pm': 'quản lý dự án',
            'sale': 'bán hàng',
            'sales': 'bán hàng',
            'tphcm': 'thành phố hồ chí minh',
            'hcm': 'thành phố hồ chí minh',
            'hn': 'hà nội',
            'đn': 'đà nẵng',
            'tmdt': 'thương mại điện tử',
            'f&b': 'nhà hàng dịch vụ ăn uống',
            'ui/ux': 'thiết kế giao diện'
        }
        words = text.lower().split()
        expanded_words = []
        for word in words:
            expanded_words.append(abbreviations.get(word, word))
        return ' '.join(expanded_words)

    def normalize_text(self, text: str) -> str:
        if not text:
            return text
        text = text.lower().strip()
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalize_location(self, location: str) -> str:
        if not location:
            return location
        location = self.normalize_text(location)
        location_patterns = {
            r'(?:tp|thành phố|t\.p|tp\.|t\.p\.)?.*?(?:hcm|ho chi minh|sai gon|sg|tphcm)': 'thành phố hồ chí minh',
            r'hcm|tphcm|sai gon|sg': 'thành phố hồ chí minh',
            r'(?:tp|thành phố|t\.p|tp\.|t\.p\.)?.*?(?:hn|ha noi|hanoi)': 'hà nội',
            r'(?:tp|thành phố|t\.p|tp\.|t\.p\.)?.*?(?:dn|da nang|danang)': 'đà nẵng',
        }
        for pattern, replacement in location_patterns.items():
            if re.search(pattern, location, re.IGNORECASE):
                return replacement
        return location

    def _update_vectors(self):
        all_corpus = []
        for industry, terms in self.corpus.items():
            all_corpus.extend(terms)
        self.vectors = self.vectorizer.fit_transform(all_corpus)

    def semantic_similarity(self, query: str, texts: List[str], threshold: float = 0.5) -> bool:
        if not query or not texts or all(not text for text in texts):
            return False
        query = self._expand_abbreviations(self.normalize_text(query))
        texts = [self._expand_abbreviations(self.normalize_text(text)) for text in texts if text]
        if not texts:
            return False
        try:
            expanded_query = self._add_synonyms(query)
            text_vectors = self.vectorizer.fit_transform([expanded_query] + texts)
            similarities = cosine_similarity(text_vectors[0:1], text_vectors[1:])[0]
            return bool(any(sim > threshold for sim in similarities))
        except Exception as e:
            logging.error(f"Error in semantic similarity: {str(e)}")
            return False

    def get_keyword_suggestions(self, term: str, keyword_dict: dict) -> List[str]:
        suggestions = []
        if not term:
            return suggestions
        term = self.normalize_text(term.lower())
        scored_suggestions = []
        for main_keyword, related_keywords in keyword_dict.items():
            main_score = fuzz.partial_ratio(term, main_keyword.lower())
            semantic_score = 0
            if self.semantic_similarity(term, [main_keyword] + related_keywords, threshold=0.5):
                semantic_score = 100
            total_score = 0.6 * main_score + 0.4 * semantic_score
            for kw in [main_keyword] + related_keywords:
                if kw.lower() != term:
                    kw_score = fuzz.partial_ratio(term, kw.lower())
                    combined_score = 0.5 * kw_score + 0.5 * total_score
                    scored_suggestions.append((kw, combined_score))
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        suggestions = [kw for kw, score in scored_suggestions if score > 40][:5]
        return list(set(suggestions))

    def combine_entities(self, entities: List[Dict], entity_type: str, message_text: str = None) -> str:
        if entity_type == 'salary':
            salary_entities = [e for e in sorted(entities, key=lambda x: x['start']) if e['entity'] == 'salary']
            if len(salary_entities) > 1 and message_text:
                start_idx = salary_entities[0]['start']
                end_idx = salary_entities[-1]['end']
                text_between = message_text[start_idx:end_idx]
                if '-' in text_between:
                    return text_between.strip()
            values = [e['value'] for e in sorted(entities, key=lambda x: x['start']) if e['entity'] == entity_type]
            return ' '.join(values) if values else None
        else:
            values = [e['value'] for e in sorted(entities, key=lambda x: x['start']) if e['entity'] == entity_type]
            if entity_type == 'specialization' and not values:
                industry_values = [e['value'] for e in entities if e['entity'] == 'industry']
                for val in industry_values:
                    if any(fuzz.partial_ratio(self.normalize_text(val.lower()), kw.lower()) > 80 
                           for kw in self.SPECIALIZATION_KEYWORDS.get('lập trình', [])):
                        return val
            return ' '.join(values) if values else None

    def reset_slots(self) -> List[Dict[Text, Any]]:
        return [
            SlotSet("industry", None),
            SlotSet("specialization", None),
            SlotSet("location", None),
            SlotSet("salary", None)
        ]

    def _get_corpus_similarity(self, query: str, threshold: float = 0.5) -> List[str]:
        if not query:
            return []
        query = self._expand_abbreviations(self.normalize_text(query))
        corpus_terms = []
        for industry, terms in self.corpus.items():
            if query in industry.lower() or any(query in kw.lower() for kw in terms):
                corpus_terms.extend(terms)
        if not corpus_terms:
            corpus_terms = [term for terms in self.corpus.values() for term in terms]
        temp_corpus = corpus_terms + [query]
        vectors = self.vectorizer.fit_transform(temp_corpus)
        query_vector = vectors[-1:]
        similarities = cosine_similarity(query_vector, vectors[:-1])[0]
        similar_keywords = []
        for idx, sim in enumerate(similarities):
            if sim > threshold:
                keywords = corpus_terms[idx].split()
                similar_keywords.extend(keywords)
        return list(set(similar_keywords))

    def calculate_relevance_score(self, row, search_terms, industry=None, corpus_terms=None):
        score = 0
        all_terms = set(search_terms)
        if corpus_terms:
            all_terms.update(corpus_terms)
        all_terms = list(all_terms)
        
        title_scores = [fuzz.partial_ratio(term, str(row['title']).lower()) for term in all_terms]
        title_score = max(title_scores) if title_scores else 0
        if any(term.lower() in str(row['title']).lower() for term in search_terms):
            score += title_score * 0.4
        else:
            score += title_score * 0.3
        
        industry_scores = []
        for term in all_terms:
            term_scores = [fuzz.partial_ratio(term, ind.strip().lower()) 
                           for ind in str(row['industryNames']).split(',')]
            if term_scores:
                industry_scores.append(max(term_scores))
        industry_score = max(industry_scores) if industry_scores else 0
        if industry and any(industry.lower() in ind.strip().lower() 
                            for ind in str(row['industryNames']).split(',')):
            score += industry_score * 0.3
        else:
            score += industry_score * 0.2
        
        semantic_score = 0
        if any(self.semantic_similarity(term, 
                                       [str(row['title']), str(row['industryNames']), str(row.get('description', ''))], 
                                       threshold=0.5) for term in all_terms):
            semantic_score = 100
        score += semantic_score * 0.3
        
        if corpus_terms:
            corpus_scores = []
            text_to_match = f"{str(row['title'])} {str(row['industryNames'])} {str(row.get('description', ''))}".lower()
            for term in corpus_terms:
                if term.lower() in text_to_match:
                    corpus_scores.append(100)
                else:
                    corpus_scores.append(fuzz.partial_ratio(term, text_to_match))
            corpus_score = max(corpus_scores) if corpus_scores else 0
            score += corpus_score * 0.2
        
        if any(term.lower() in str(row['title']).lower() or 
               term.lower() in str(row['industryNames']).lower() for term in search_terms):
            score += 50
        
        return min(score, 100)

    def find_exact_matches(self, df, search_term, industry_keywords):
        if not search_term:
            return df, False
        search_term = self.normalize_text(search_term.lower())
        exact_matches = df[
            (df['title'].str.lower().str.contains(search_term, na=False)) | 
            (df['industryNames'].str.lower().str.contains(search_term, na=False))
        ]
        if not exact_matches.empty:
            return exact_matches, True
        fuzzy_matches = df[
            df.apply(
                lambda row: 
                    any(fuzz.partial_ratio(search_term, kw.lower()) > 80
                        for main_key, keywords in industry_keywords.items()
                        for kw in [main_key] + keywords),
                axis=1
            )
        ]
        return fuzzy_matches, not fuzzy_matches.empty

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logging.debug(f"Processing query with slots: {tracker.slots}")
        entities = tracker.latest_message.get('entities', [])
        message_text = tracker.latest_message.get('text', '')
        
        new_slots = {}
        for entity_type in ['industry', 'specialization', 'location', 'salary']:
            value = self.combine_entities(entities, entity_type, message_text)
            if value:
                new_slots[entity_type] = value
            else:
                new_slots[entity_type] = tracker.get_slot(entity_type)
        
        logging.debug(f"Initial new_slots: {new_slots}")
        
        # Ánh xạ ngành nghề và chuyên môn
        if new_slots['industry']:
            industry_values = new_slots['industry'].split()
            for val in industry_values:
                if any(fuzz.partial_ratio(self.normalize_text(val.lower()), kw.lower()) > 80 
                       for kw in self.SPECIALIZATION_KEYWORDS.get('lập trình', [])):
                    new_slots['specialization'] = val
                    new_slots['industry'] = 'it phần mềm'
                    break
            else:
                for main_key, keywords in self.INDUSTRY_KEYWORDS.items():
                    if any(fuzz.partial_ratio(self.normalize_text(new_slots['industry'].lower()), kw.lower()) > 80 for kw in [main_key] + keywords):
                        new_slots['industry'] = main_key
                        break
        if new_slots['specialization']:
            for main_key, keywords in self.SPECIALIZATION_KEYWORDS.items():
                if any(fuzz.partial_ratio(self.normalize_text(new_slots['specialization'].lower()), kw.lower()) > 80 for kw in [main_key] + keywords):
                    new_slots['specialization'] = main_key
                    break
        
        logging.debug(f"Final new_slots: {new_slots}")
        
        try:
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "job_post.csv")
            if not os.path.exists(csv_path):
                alt_csv_path = "d:/JobRadar_/job_post.csv"
                if not os.path.exists(alt_csv_path):
                    dispatcher.utter_message(text="Xin lỗi, không tìm thấy dữ liệu việc làm.")
                    return self.reset_slots()
                csv_path = alt_csv_path
            
            df = pd.read_csv(csv_path)
            filtered_df = df.copy()
            final_results = filtered_df.copy()
            suggestions = {'industry': [], 'specialization': []}

            if new_slots['salary']:
                salary_value = str(new_slots['salary'])
                try:
                    if '-' in salary_value:
                        min_salary, max_salary = salary_value.split('-')
                        min_salary_number = float(''.join(filter(str.isdigit, min_salary.strip())))
                        max_salary_number = float(''.join(filter(str.isdigit, max_salary.strip())))
                        min_target_salary = min_salary_number * 1000000
                        max_target_salary = max_salary_number * 1000000
                        salary_values = filtered_df['salary'].astype(float)
                        salary_filter = (salary_values >= min_target_salary) & (salary_values <= max_target_salary)
                        filtered_df = filtered_df[salary_filter]
                    else:
                        salary_number = float(''.join(filter(str.isdigit, salary_value)))
                        target_salary = salary_number * 1000000
                        salary_values = filtered_df['salary'].astype(float)
                        salary_filter = salary_values >= target_salary
                        filtered_df = filtered_df[salary_filter]
                    
                    if filtered_df.empty:
                        dispatcher.utter_message(text=f"Không tìm thấy việc làm nào có mức lương từ {salary_value}.")
                        relaxed_salary = salary_number * 0.8 * 1000000
                        salary_filter = filtered_df['salary'].astype(float) >= relaxed_salary
                        relaxed_results = filtered_df[salary_filter].copy()
                        if not relaxed_results.empty:
                            dispatcher.utter_message(text=f"Tuy nhiên, đây là một số công việc với mức lương từ {salary_number * 0.8:.1f} triệu:")
                            final_results = relaxed_results
                    else:
                        final_results = filtered_df.copy()
                except (ValueError, AttributeError) as e:
                    logging.error(f"Error processing salary: {e}")
                    dispatcher.utter_message(text=f"Không thể xử lý mức lương: {salary_value}")
                    return self.reset_slots()

            if new_slots['industry']:
                logging.debug(f"Searching for industry: {new_slots['industry']}")
                negative_keywords = self.NEGATIVE_KEYWORDS.get(new_slots['industry'].lower(), [])
                exact_matches, found_exact = self.find_exact_matches(
                    final_results, 
                    new_slots['industry'],
                    self.INDUSTRY_KEYWORDS
                )
                if found_exact:
                    final_results = exact_matches
                else:
                    corpus_terms = self._get_corpus_similarity(new_slots['industry'], threshold=0.5)
                    final_results['relevance_score'] = final_results.apply(
                        lambda row: self.calculate_relevance_score(
                            row, 
                            [new_slots['industry']], 
                            new_slots['industry'],
                            corpus_terms
                        ), 
                        axis=1
                    )
                    final_results = final_results[final_results['relevance_score'] > 50]
                
                final_results = final_results[
                    ~final_results.apply(
                        lambda row: any(
                            kw.lower() in f"{str(row['title']).lower()} {str(row['industryNames']).lower()}"
                            for kw in negative_keywords
                        ),
                        axis=1
                    )
                ]
                suggestions['industry'] = self.get_keyword_suggestions(new_slots['industry'], self.INDUSTRY_KEYWORDS)

            if new_slots['location']:
                location = self.normalize_location(new_slots['location'])
                location_filter = final_results['cityName'].str.lower().apply(
                    lambda x: self.normalize_location(str(x)).lower() == location.lower()
                )
                final_results = final_results[location_filter]
                if final_results.empty:
                    dispatcher.utter_message(text=f"Không tìm thấy công việc ở {new_slots['location']}. Dưới đây là một số công việc tương tự ở các địa điểm khác:")
                    final_results = filtered_df.copy()
                    if new_slots['industry']:
                        final_results['relevance_score'] = final_results.apply(
                            lambda row: self.calculate_relevance_score(
                                row, [new_slots['industry']], new_slots['industry'], None
                            ), axis=1
                        )
                        final_results = final_results[final_results['relevance_score'] > 50]

            if new_slots['specialization']:
                specialization_filter = final_results.apply(
                    lambda row: self.semantic_similarity(
                        new_slots['specialization'], 
                        [str(row['title']), str(row.get('description', ''))],
                        threshold=0.5
                    ) or new_slots['specialization'].lower() in str(row['title']).lower(),
                    axis=1
                )
                final_results = final_results[specialization_filter]
                suggestions['specialization'] = self.get_keyword_suggestions(
                    new_slots['specialization'], 
                    self.SPECIALIZATION_KEYWORDS
                )

            if final_results.empty:
                dispatcher.utter_message(text="Không tìm thấy việc làm nào phù hợp với yêu cầu của bạn.")
                if suggestions['industry'] or suggestions['specialization']:
                    sugg_text = "Bạn có thể thử các từ khóa: "
                    if suggestions['industry']:
                        sugg_text += f"Ngành: {', '.join(suggestions['industry'])}. "
                    if suggestions['specialization']:
                        sugg_text += f"Chuyên môn: {', '.join(suggestions['specialization'])}."
                    dispatcher.utter_message(text=sugg_text)
                return self.reset_slots()

            if 'relevance_score' not in final_results.columns:
                final_results['relevance_score'] = final_results.apply(
                    lambda row: self.calculate_relevance_score(
                        row, [], None, None
                    ),
                    axis=1
                )

            final_results = final_results.sort_values('relevance_score', ascending=False).head(7)

            jobs = []
            for _, row in final_results.iterrows():
                jobs.append({
                    "title": str(row.get('title', 'N/A')),
                    "cityName": str(row.get('cityName', 'N/A')),
                    "salary": float(row.get('salary', 0)),
                    "companyName": str(row.get('companyName', '')),
                    "logo": str(row.get('logo', '')),
                    "industryNames": str(row.get('industryNames', 'N/A')),
                    "postId": str(row.get('postId', '')),
                    "job_url": f"http://localhost:3000/jobs/job-detail/{row.get('postId', '')}" if row.get('postId') else None
                })
            dispatcher.utter_message(json_message={"jobs": jobs})
            # Giữ slot nếu tìm kiếm thành công
            return [
                SlotSet("industry", new_slots['industry']),
                SlotSet("specialization", new_slots['specialization']),
                SlotSet("location", new_slots['location']),
                SlotSet("salary", new_slots['salary'])
            ]

        except Exception as e:
            logging.error(f"Error in job search: {str(e)}")
            dispatcher.utter_message(text=f"Không thể tìm thấy công việc, vui lòng tìm kiếm một nội dung khác")
            return self.reset_slots()

    def _check_salary(self, job_salary: str, target_salary: str) -> bool:
        if not job_salary or not target_salary:
            return False
        try:
            job_salary = str(job_salary).lower().strip()
            target_salary = str(target_salary).lower().strip()
            target_num = float(''.join(filter(str.isdigit, target_salary)))
            target_value = target_num * 1000000
            if job_salary.isdigit():
                return float(job_salary) >= target_value
            if '-' in job_salary:
                salary_parts = job_salary.split('-')
                max_salary = float(''.join(filter(str.isdigit, salary_parts[-1].strip())))
                return max_salary * 1e6 >= target_value
            if 'triệu' in job_salary:
                salary_num = float(''.join(filter(str.isdigit, job_salary)))
                return salary_num * 1e6 >= target_value
            return False
        except ValueError:
            return False

class ActionClearSlots(Action):
    def name(self) -> Text:
        return "action_clear_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Đã xóa tất cả tiêu chí tìm kiếm.")
        return [
            SlotSet("industry", None),
            SlotSet("specialization", None),
            SlotSet("location", None),
            SlotSet("salary", None)
        ]

class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(response="utter_default")
        return []