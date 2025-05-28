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


class ActionSearchJobs(Action):

    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            stop_words=['và', 'hoặc', 'của', 'với', 'các', 'những']
        )
        # Khởi tạo corpus ban đầu với các từ khóa cơ bản
        self.corpus = self._initialize_corpus()
        self.vectors = None
        self._update_vectors()
        # Định nghĩa từ khóa cho ngành và chuyên môn
        self.INDUSTRY_KEYWORDS = {
            'thương mại điện tử': ['thương mại điện tử', 'bán hàng trực tuyến', 'e-commerce', 'sàn thương mại', 'mua sắm online'],
            'marketing': ['marketing', 'truyền thông', 'quảng cáo', 'tiếp thị', 'thương hiệu', 'pr', 'digital marketing'],
            'it phần cứng': ['it', 'kỹ thuật phần cứng', 'công nghệ phần cứng', 'hardware', 'network engineering'],
            'công nghệ ô tô': ['công nghệ ô tô', 'kỹ thuật ô tô', 'sửa chữa ô tô', 'automotive technology'],
            'it phần mềm': ['it', 'lập trình', 'phát triển phần mềm', 'công nghệ thông tin', 'software development', 'cntt'],
            'nhà hàng': ['nhà hàng', 'khách sạn', 'dịch vụ ăn uống', 'hospitality', 'f&b'],
            'thiết kế': ['thiết kế', 'in ấn', 'đồ họa', 'graphic design', 'ui/ux'],
            'cơ khí': ['cơ khí', 'điện - điện tử', 'mechanical engineering', 'electrical engineering', "điện tử"],
            'kinh doanh': ['kinh doanh', 'bán hàng', 'thương mại', 'business', 'sales'],
            'giáo dục': ['giáo dục', 'đào tạo', 'giảng dạy', 'education', 'teaching'],
            'kiến trúc': ['kiến trúc', 'xây dựng', 'công trình', 'architecture', 'construction'],
            'tài chính': ['tài chính', 'ngân hàng', 'đầu tư', 'finance', 'banking'],
            'viễn thông': ['viễn thông', 'mạng viễn thông', 'telecommunications', 'telecom'],
            'y tế': ['y tế', 'chăm sóc sức khỏe', 'điều dưỡng', 'healthcare', 'medical'],
            'logistics': ['logistics', 'vận tải', 'chuỗi cung ứng', 'supply chain', 'warehousing'],
            'kế toán': ['kế toán', 'kiểm toán', 'báo cáo tài chính', 'accounting', 'auditing'],
            'sản xuất': ['sản xuất', 'chế tạo', 'công nghiệp', 'manufacturing', 'production'],
            'tài xế': ['tài xế', 'lái xe', 'vận chuyển', 'driver'],
            'luật': ['luật', 'pháp lý', 'tư vấn pháp luật', 'law', 'legal'],
            'phiên dịch': ['phiên dịch', 'dịch thuật', 'thông dịch', 'translation', 'interpretation'],
            'hệ thống nhúng': ['hệ thống nhúng', 'iot', 'internet vạn vật', 'embedded systems', 'internet of things'],
        }
        self.SPECIALIZATION_KEYWORDS = {
            'lập trình': ['developer', 'programmer', 'software engineer', 'frontend', 'backend', 'fullstack', 'devops', 'lập trình viên', 'kỹ sư phần mềm', 'phát triển phần mềm'],
            'quảng cáo': ['quảng cáo', 'content marketing', 'copywriting', 'social media', 'seo', 'sem', 'digital marketing', 'tiếp thị nội dung', 'quản lý truyền thông'],
            'thiết kế': ['thiết kế', 'graphic designer', 'ui/ux', 'illustrator', 'photoshop', 'thiết kế đồ họa', 'thiết kế giao diện', 'ux designer', 'ui designer'],
            'kỹ thuật': ['kỹ thuật', 'engineer', 'technician', 'maintenance', 'kỹ sư', 'bảo trì', 'kỹ thuật viên'],
            'bán hàng': ['bán hàng', 'sales', 'business development', 'crm', 'đại diện bán hàng', 'phát triển kinh doanh', 'quản lý khách hàng'],
            'giáo viên': ['giáo viên', 'giảng viên', 'teacher', 'instructor', 'tutor', 'trợ giảng', 'đào tạo'],
            'kiến trúc sư': ['kiến trúc sư', 'architect', 'civil engineer', 'cad', 'kỹ sư xây dựng', 'thiết kế công trình'],
            'tài chính': ['phân tích tài chính', 'financial analyst', 'investment', 'credit', 'quản lý tài chính', 'tư vấn đầu tư', 'phân tích tín dụng'],
            'điều dưỡng': ['điều dưỡng', 'y tá', 'nurse', 'healthcare', 'chăm sóc sức khỏe', 'hộ lý'],
            'tài xế': ['tài xế', 'lái xe', 'driver', 'vận chuyển', 'giao hàng'],
            'luật sư': ['luật sư', 'pháp chế', 'legal counsel', 'tư vấn pháp luật', 'cố vấn pháp lý'],
            'phiên dịch': ['phiên dịch', 'biên dịch', 'translator', 'interpreter', 'thông dịch', 'dịch thuật'],
            'iot': ['iot', 'embedded', 'firmware', 'microcontroller', 'hệ thống nhúng', 'internet vạn vật'],
            'phân tích dữ liệu': ['data analyst', 'phân tích dữ liệu', 'business intelligence', 'data scientist', 'big data', 'sql', 'python'],
            'quản lý dự án': ['project manager', 'quản lý dự án', 'pm', 'scrum master', 'agile', 'quản lý tiến độ'],
            'kiểm thử phần mềm': ['qa', 'quality assurance', 'tester', 'kiểm thử phần mềm', 'software testing', 'test engineer'],
            'hỗ trợ khách hàng': ['hỗ trợ khách hàng', 'customer support', 'customer service'],
            'an ninh mạng': ['cybersecurity', 'an ninh mạng', 'network security', 'information security', 'ethical hacker'],
            'nhân sự': ['human resources', 'nhân sự', 'hr', 'recruitment', 'tuyển dụng', 'talent acquisition'],
            'logistics': ['logistics', 'vận hành kho', 'warehouse', 'supply chain', 'chuỗi cung ứng', 'quản lý vận tải'],
            'chăm sóc sức khỏe': ['healthcare', 'chăm sóc sức khỏe', 'bác sĩ', 'doctor', 'y sĩ', 'physician'],
            'kế toán': ['kế toán', 'accountant', 'bookkeeper', 'báo cáo tài chính', 'financial reporting'],
        }

    def name(self) -> Text:
        return "action_search_jobs"

    def _initialize_corpus(self):
        """Khởi tạo corpus với các từ khóa cơ bản"""
        basic_terms = [
            "thương mại điện tử ecommerce tmdt online shopping marketplace sàn bán hàng online digital commerce electronic commerce",
            "marketing truyền thông quảng cáo pr public relations digital marketing social media content marketing brand marketing tiếp thị truyền thông đại chúng marketing online seo sem",
            "it phần cứng hardware network engineering mạng máy tính thiết bị phần cứng computer hardware network administrator system engineer hạ tầng mạng thiết bị mạng network infrastructure server phần cứng máy tính thiết bị ngoại vi",
            "công nghệ ô tô kỹ thuật ô tô automotive technology car technology automotive engineering xe hơi sửa chữa ô tô bảo dưỡng xe",
            "it phần mềm công nghệ thông tin lập trình viên developer programmer software engineer frontend backend fullstack devops mobile app web development coding programming cntt phát triển phần mềm software development",
            "nhà hàng khách sạn hospitality restaurant hotel food beverage f&b dịch vụ ăn uống quản lý nhà hàng quản lý khách sạn chef đầu bếp",
            "thiết kế in ấn graphic design ui ux designer printing nghệ thuật đồ họa mỹ thuật thiết kế đồ họa thiết kế web thiết kế sản phẩm industrial design",
            "cơ khí điện điện tử mechanical engineering electrical engineering electronic kỹ sư cơ khí kỹ sư điện tự động hóa automation engineering",
            "kinh doanh bán hàng sales business development thương mại marketing bán lẻ retail thương mại quốc tế trading import export xuất nhập khẩu",
            "giáo dục đào tạo education teaching teacher instructor giảng viên giáo viên training coach mentoring tư vấn giáo dục education consultant",
            "kiến trúc xây dựng architecture construction civil engineering thiết kế kiến trúc construction management quản lý xây dựng building design",
            "tài chính ngân hàng finance banking investment chứng khoán securities financial analyst đầu tư tư vấn tài chính wealth management quản lý đầu tư",
            "viễn thông telecommunications telecom công nghệ viễn thông mạng viễn thông telecommunication engineering network infrastructure",
            "y tế healthcare medical bác sĩ doctor nurse điều dưỡng dược sĩ pharmacist medical care bệnh viện hospital clinic phòng khám",
            "logistics chuỗi cung ứng supply chain warehouse kho vận quản lý kho vận tải transportation shipping freight forwarding xuất nhập khẩu",
            "kế toán kiểm toán accounting audit accountant auditor financial reporting báo cáo tài chính tax thuế bookkeeping",
            "sản xuất manufacturing production nhà máy factory operation management quản lý sản xuất quality control kiểm soát chất lượng",
            "tài xế lái xe driver chauffeur vận chuyển transportation delivery giao hàng shipper xe tải xe khách",
            "luật pháp lý legal law lawyer attorney tư vấn pháp luật legal consultant legal advisor luật sư tư vấn luật",
            "phiên dịch biên dịch translation interpreter translator ngôn ngữ language dịch thuật thông dịch viên",
            "hệ thống nhúng iot internet of things embedded systems firmware microcontroller vi điều khiển internet vạn vật smart devices thiết bị thông minh sensor cảm biến"
        ]
        return basic_terms

    def _update_vectors(self):
        """Cập nhật vectors cho corpus"""
        self.vectors = self.vectorizer.fit_transform(self.corpus)

    def semantic_similarity(self, query: str, texts: List[str], threshold: float = 0.25) -> bool:
        """Tính độ tương đồng ngữ nghĩa giữa query và list các text"""
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
            print(f"Error in semantic similarity: {str(e)}")
            return False

    def _expand_abbreviations(self, text: str) -> str:
        """Mở rộng các từ viết tắt phổ biến"""
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
            'tphcm': 'hồ chí minh',
            'hcm': 'hồ chí minh',
            'hn': 'hà nội',
            'đn': 'đà nẵng'
        }
        
        words = text.lower().split()
        expanded_words = []
        for word in words:
            if word in abbreviations:
                expanded_words.append(abbreviations[word])
            else:
                expanded_words.append(word)
        return ' '.join(expanded_words)

    def _add_synonyms(self, text: str) -> str:
        """Thêm các từ đồng nghĩa vào query để tăng khả năng match"""
        synonyms = {
            'công nghệ thông tin': 'it phần mềm lập trình developer programmer coding software cntt kỹ sư phần mềm',
            'lập trình': 'developer programmer coding software engineering code viết mã kỹ sư phần mềm',
            'kinh doanh': 'sales bán hàng thương mại business thương mại hóa phát triển kinh doanh',
            'nhân sự': 'human resources hr tuyển dụng recruitment quản lý nhân sự quản trị nhân lực',
            'kế toán': 'accounting accountant finance tài chính kiểm toán báo cáo tài chính',
            'tiếp thị': 'marketing quảng cáo advertising truyền thông pr digital marketing',
            'thiết kế': 'design graphic ui ux đồ họa mỹ thuật sáng tạo giao diện',
            'quản lý': 'manager management leader quản trị điều hành giám đốc',
            'giáo dục': 'education teaching giảng dạy đào tạo giáo viên giảng viên',
            'y tế': 'healthcare medical bác sĩ nurse điều dưỡng chăm sóc sức khỏe y khoa',
            'xây dựng': 'construction kiến trúc công trình kỹ sư xây dựng thi công',
            'logistics': 'chuỗi cung ứng supply chain vận tải kho bãi hậu cần',
            'luật': 'pháp lý legal luật sư tư vấn pháp luật cố vấn pháp lý',
            'phiên dịch': 'dịch thuật translation thông dịch interpreter ngôn ngữ',
            'tài chính': 'finance banking investment ngân hàng đầu tư chứng khoán quỹ đầu tư tư vấn tài chính phân tích tài chính financial analyst'
        }
        
        words = text.lower().split()
        expanded_words = words.copy()
        for word in words:
            if word in synonyms:
                expanded_words.extend(synonyms[word].split())
        return ' '.join(set(expanded_words))

    def normalize_text(self, text: str) -> str:
        """Chuẩn hóa text để tìm kiếm thông minh hơn"""
        if not text:
            return text
        
        text = text.lower().strip()
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalize_location(self, location: str) -> str:
        """Chuẩn hóa địa điểm với xử lý thông minh hơn"""
        if not location:
            return location

        location = self.normalize_text(location)
        
        location_patterns = {
            r'(?:tp|thành phố|t\.p|tp\.|t\.p\.)?.*?(?:hcm|ho chi minh|sai gon|sg|tphcm)': 'thành phố hồ chí minh',
            r'(?:tp|thành phố|t\.p|tp\.|t\.p\.)?.*?(?:hn|ha noi|hanoi)': 'hà nội',
            r'(?:tp|thành phố|t\.p|tp\.|t\.p\.)?.*?(?:dn|da nang|danang)': 'đà nẵng',
        }
        
        for pattern, replacement in location_patterns.items():
            if re.search(pattern, location):
                return replacement
                
        return location

    def get_keyword_suggestions(self, term: str, keyword_dict: dict) -> List[str]:
        """Tìm các từ khóa gợi ý dựa trên term với xếp hạng độ chính xác"""
        suggestions = []
        if not term:
            return suggestions
            
        term = self.normalize_text(term.lower())
        scored_suggestions = []
        
        for main_keyword, related_keywords in keyword_dict.items():
            # Tính điểm fuzzy matching cho từ khóa chính
            main_score = fuzz.partial_ratio(term, main_keyword.lower())
            # Tính điểm semantic similarity
            semantic_score = 0
            if self.semantic_similarity(term, [main_keyword] + related_keywords):
                semantic_score = 100  # Gán điểm cao nếu có semantic match
                
            # Tổng hợp điểm
            total_score = 0.6 * main_score + 0.4 * semantic_score
            
            # Thêm các từ khóa liên quan
            for kw in [main_keyword] + related_keywords:
                if kw.lower() != term:
                    kw_score = fuzz.partial_ratio(term, kw.lower())
                    combined_score = 0.5 * kw_score + 0.5 * total_score
                    scored_suggestions.append((kw, combined_score))
        
        # Sắp xếp và lấy top 5
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        suggestions = [kw for kw, score in scored_suggestions if score > 50][:5]
        return list(set(suggestions))

    def get_matching_keywords(self, input_text: str, keyword_dict: dict) -> List[Dict]:
        """Tìm các từ khóa phù hợp dựa trên input"""
        matches = []
        input_text = self.normalize_text(input_text.lower())
        
        for main_keyword, related_keywords in keyword_dict.items():
            if input_text in main_keyword.lower() or any(input_text in kw.lower() for kw in related_keywords):
                matches.append({
                    'main': main_keyword,
                    'related': related_keywords
                })
            else:
                # Kiểm tra fuzzy matching
                for kw in [main_keyword] + related_keywords:
                    if fuzz.partial_ratio(input_text, kw.lower()) > 80:
                        matches.append({
                            'main': main_keyword,
                            'related': related_keywords
                        })
                        break
        return matches

    def combine_entities(self, entities: List[Dict], entity_type: str, message_text: str = None) -> str:
        """Kết hợp các entity cùng loại thành một chuỗi hoàn chỉnh"""
        if entity_type == 'salary':
            # Tìm tất cả các entity salary
            salary_entities = [e for e in sorted(entities, key=lambda x: x['start']) if e['entity'] == 'salary']
            if len(salary_entities) > 1:
                # Nếu có nhiều hơn 1 entity salary, kiểm tra xem có phải khoảng lương không
                if message_text:
                    start_idx = salary_entities[0]['start']
                    end_idx = salary_entities[-1]['end']
                    text_between = message_text[start_idx:end_idx]
                    if '-' in text_between:
                        # Nếu có dấu '-' giữa các entity salary, đây là khoảng lương
                        return text_between.strip()
            
            # Nếu chỉ có 1 entity hoặc không phải khoảng lương
            values = []
            for entity in sorted(entities, key=lambda x: x['start']):
                if entity['entity'] == entity_type:
                    values.append(entity['value'])
            return ' '.join(values) if values else None
        else:
            # Xử lý các entity khác như cũ
            values = []
            for entity in sorted(entities, key=lambda x: x['start']):
                if entity['entity'] == entity_type:
                    values.append(entity['value'])
            return ' '.join(values) if values else None

    def reset_slots(self) -> List[Dict[Text, Any]]:
        """Reset tất cả các slot về None"""
        return [
            SlotSet("industry", None),
            SlotSet("specialization", None),
            SlotSet("location", None),
            SlotSet("salary", None)
        ]

    def _get_corpus_similarity(self, query: str, threshold: float = 0.3) -> List[str]:
        """Tìm các từ khóa tương tự từ corpus dựa trên query"""
        if not query:
            return []
            
        # Chuẩn hóa query
        query = self._expand_abbreviations(self.normalize_text(query))
        
        # Thêm query vào corpus tạm thời để vectorize
        temp_corpus = self.corpus + [query]
        vectors = self.vectorizer.fit_transform(temp_corpus)
        
        # Tính similarity giữa query và tất cả các entry trong corpus
        query_vector = vectors[-1:]  # Vector của query (entry cuối cùng)
        similarities = cosine_similarity(query_vector, vectors[:-1])[0]  # So sánh với tất cả trừ chính nó
        
        # Lấy các từ khóa từ corpus có độ tương đồng cao
        similar_keywords = []
        for idx, sim in enumerate(similarities):
            if sim > threshold:
                # Tách các từ khóa từ corpus entry
                keywords = self.corpus[idx].split()
                similar_keywords.extend(keywords)
        
        return list(set(similar_keywords))  # Loại bỏ trùng lặp

    def calculate_relevance_score(self, row, search_terms, industry=None, corpus_terms=None):
        """Tính điểm độ chính xác cho một công việc"""
        score = 0
        
        # Kết hợp search_terms với corpus_terms
        all_terms = set(search_terms)
        if corpus_terms:
            all_terms.update(corpus_terms)
        all_terms = list(all_terms)
        
        # Điểm cho tên công việc (35%)
        title_scores = [fuzz.partial_ratio(term, str(row['title']).lower()) for term in all_terms]
        title_score = max(title_scores) if title_scores else 0
        score += title_score * 0.35
        
        # Điểm cho ngành (25%)
        industry_scores = []
        for term in all_terms:
            term_scores = [fuzz.partial_ratio(term, ind.strip().lower()) 
                         for ind in str(row['industryNames']).split(',')]
            if term_scores:
                industry_scores.append(max(term_scores))
        industry_score = max(industry_scores) if industry_scores else 0
        score += industry_score * 0.25
        
        # Điểm cho semantic similarity (25%)
        semantic_score = 0
        if any(self.semantic_similarity(term, 
            [str(row['title']), str(row['industryNames'])], 
            threshold=0.3) for term in all_terms):
            semantic_score = 100
        score += semantic_score * 0.25
        
        # Điểm cho corpus match (15%)
        if corpus_terms:
            corpus_scores = []
            text_to_match = f"{str(row['title'])} {str(row['industryNames'])}".lower()
            for term in corpus_terms:
                if term.lower() in text_to_match:
                    corpus_scores.append(100)
                else:
                    corpus_scores.append(fuzz.partial_ratio(term, text_to_match))
            corpus_score = max(corpus_scores) if corpus_scores else 0
            score += corpus_score * 0.15
        
        # Bonus điểm cho exact match với industry
        if industry and any(industry.lower() in ind.strip().lower() 
            for ind in str(row['industryNames']).split(',')):
            score += 50
            
        return min(score, 100)  # Giới hạn điểm tối đa là 100

    def find_exact_matches(self, df, search_term, industry_keywords):
        """Tìm các kết quả match chính xác với từ khóa"""
        if not search_term:
            return df, False

        search_term = self.normalize_text(search_term.lower())
        
        # Tìm trong title và industryNames
        exact_matches = df[
            df.apply(lambda row: 
                search_term in str(row['title']).lower() or
                search_term in str(row['industryNames']).lower() or
                # Kiểm tra trong các từ khóa của ngành
                any(search_term in kw.lower() 
                    for main_key, keywords in industry_keywords.items()
                    for kw in [main_key] + keywords
                    if fuzz.partial_ratio(search_term, kw.lower()) > 80),
                axis=1
            )
        ]
        
        return exact_matches, not exact_matches.empty

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Lấy entities từ message cuối cùng
        entities = tracker.latest_message.get('entities', [])
        message_text = tracker.latest_message.get('text', '')
        
        # Tạo dictionary chứa các slot mới
        new_slots = {}
        for entity_type in ['industry', 'specialization', 'location', 'salary']:
            value = self.combine_entities(entities, entity_type, message_text)
            if value:
                new_slots[entity_type] = value
            else:
                new_slots[entity_type] = tracker.get_slot(entity_type)
        
        # Đọc dữ liệu từ CSV
        try:
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "job_post.csv")
            if not os.path.exists(csv_path):
                alt_csv_path = "d:/JobRadar_/job_post.csv"
                if not os.path.exists(alt_csv_path):
                    dispatcher.utter_message(text="Xin lỗi, không tìm thấy dữ liệu việc làm.")
                    return []
                csv_path = alt_csv_path
            
            df = pd.read_csv(csv_path)
            filtered_df = df.copy()
            final_results = filtered_df.copy()
            suggestions = {
                'industry': [],
                'specialization': []
            }

            if new_slots['salary']:
                # Xử lý giá trị lương từ slot
                salary_value = str(new_slots['salary'])
                try:
                    # Kiểm tra xem có phải khoảng lương không (VD: "10 - 15 triệu")
                    if '-' in salary_value:
                        # Tách thành 2 giá trị
                        min_salary, max_salary = salary_value.split('-')
                        # Lấy số từ chuỗi cho lương tối thiểu và tối đa
                        min_salary_number = float(''.join(filter(str.isdigit, min_salary.strip())))
                        max_salary_number = float(''.join(filter(str.isdigit, max_salary.strip())))
                        # Chuyển thành giá trị đầy đủ
                        min_target_salary = min_salary_number * 1000000
                        max_target_salary = max_salary_number * 1000000
                        
                        # Lọc các công việc có mức lương trong khoảng
                        salary_values = filtered_df['salary'].astype(float)
                        salary_filter = (salary_values >= min_target_salary) & (salary_values <= max_target_salary)
                        filtered_df = filtered_df[salary_filter]
                    else:
                        # Xử lý giá trị lương đơn
                        salary_number = float(''.join(filter(str.isdigit, salary_value)))
                        target_salary = salary_number * 1000000
                        salary_values = filtered_df['salary'].astype(float)
                        salary_filter = salary_values >= target_salary
                        filtered_df = filtered_df[salary_filter]
                    
                    if filtered_df.empty:
                        if '-' in salary_value:
                            dispatcher.utter_message(text=f"Xin lỗi, không tìm thấy việc làm nào có mức lương từ {min_salary.strip()} đến {max_salary.strip()}.")
                        else:
                            dispatcher.utter_message(text=f"Xin lỗi, không tìm thấy việc làm nào có mức lương từ {salary_value} trở lên.")
                        return self.reset_slots()
                    final_results = filtered_df.copy()
                except (ValueError, AttributeError) as e:
                    print(f"Error processing salary: {e}")  # Debug log
                    dispatcher.utter_message(text=f"Xin lỗi, không thể xử lý mức lương: {salary_value}")
                    return self.reset_slots()

            if new_slots['industry']:
                # Thử tìm match chính xác trước
                exact_matches, found_exact = self.find_exact_matches(
                    final_results, 
                    new_slots['industry'],
                    self.INDUSTRY_KEYWORDS
                )

                if found_exact:
                    final_results = exact_matches
                    # Tính relevance_score cho exact matches
                    final_results['relevance_score'] = final_results.apply(
                        lambda row: self.calculate_relevance_score(
                            row, 
                            [new_slots['industry']], 
                            new_slots['industry'],
                            None
                        ), 
                        axis=1
                    )
                else:
                    corpus_terms = self._get_corpus_similarity(new_slots['industry'], threshold=0.3)
                    # Tính relevance_score cho tất cả các kết quả
                    final_results['relevance_score'] = final_results.apply(
                        lambda row: self.calculate_relevance_score(
                            row, 
                            [new_slots['industry']], 
                            new_slots['industry'],
                            corpus_terms
                        ), 
                        axis=1
                    )

            if new_slots['location']:
                location = self.normalize_location(new_slots['location'])
                location_filter = final_results['cityName'].str.lower().apply(lambda x: 
                    self.normalize_location(str(x)).lower() == location.lower()
                )
                final_results = final_results[location_filter]

            if new_slots['specialization']:
                specialization_filter = final_results.apply(
                    lambda row: self.semantic_similarity(
                        new_slots['specialization'], 
                        [str(row['title']), str(row.get('description', ''))]
                    ), 
                    axis=1
                )
                final_results = final_results[specialization_filter]
                suggestions['specialization'] = self.get_keyword_suggestions(
                    new_slots['specialization'], 
                    self.SPECIALIZATION_KEYWORDS
                )

            if final_results.empty:
                dispatcher.utter_message(text="Xin lỗi, không tìm thấy việc làm nào phù hợp với yêu cầu của bạn.")
                return self.reset_slots()

            # Đảm bảo có cột relevance_score cho tất cả kết quả
            if 'relevance_score' not in final_results.columns:
                final_results['relevance_score'] = final_results.apply(
                    lambda row: self.calculate_relevance_score(
                        row,
                        [],  # Không có từ khóa tìm kiếm
                        None,  # Không có ngành cụ thể
                        None  # Không có corpus terms
                    ),
                    axis=1
                )

            # Sắp xếp theo độ tương đồng và lấy top 5
            final_results = final_results.sort_values('relevance_score', ascending=False).head(5)

            # Trả về kết quả dạng JSON cho frontend
            jobs = []
            for _, row in final_results.iterrows():
                jobs.append({
                    "title": row['title'],
                    "cityName": row['cityName'],
                    "salary": row['salary'],
                    "companyName": row['companyName'],
                    "logo": row.get('logo', ''),
                    "industryNames": row['industryNames'],
                    "postId": row.get('postId', None),
                    "job_url": f"http://localhost:3000/jobs/job-detail/{row['postId']}" if 'postId' in row else None
                })
            dispatcher.utter_message(json_message={"jobs": jobs})
            return self.reset_slots()
            
        except Exception as e:
            dispatcher.utter_message(text=f"Xin lỗi, đã xảy ra lỗi khi tìm kiếm việc làm: {str(e)}")
            # Reset slots khi có lỗi
            return self.reset_slots()

    def _check_salary_match(self, job_salary: str, target_salary: float) -> bool:
        """Kiểm tra xem mức lương của công việc có phù hợp với yêu cầu không"""
        if not job_salary or not isinstance(job_salary, str):
            return False
            
        try:
            # Xử lý chuỗi lương từ database
            job_salary = str(job_salary).lower().strip()
            
            # Trường hợp lương dạng số thuần túy (15000000)
            if job_salary.isdigit():
                return float(job_salary) >= target_salary
                
            # Trường hợp có dấu gạch ngang (15000000-20000000)
            if '-' in job_salary:
                salary_parts = job_salary.split('-')
                if len(salary_parts) == 2 and salary_parts[1].strip().isdigit():
                    max_salary = float(salary_parts[1].strip())
                    return max_salary >= target_salary
                    
            # Trường hợp có chữ "triệu" 
            if 'triệu' in job_salary:
                # Lấy số từ chuỗi
                salary_num = float(''.join(filter(str.isdigit, job_salary)))
                # Chuyển về đơn vị đồng
                job_salary_value = salary_num * 1000000
                return job_salary_value >= target_salary
                
            return False
        except (ValueError, AttributeError):
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