import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(catalog_file):
        return []
    
    try:
        with open(catalog_file, "r", encoding="utf-8") as f:
            products = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
    
    # Lọc theo category và max_price
    results = [
        p for p in products
        if p.get("category") == category and p.get("price_vnd", float("inf")) <= max_price
    ]
    return results


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    today = datetime.now().strftime("%Y%m%d")
    
    # Load existing tickets
    tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        try:
            with open(tickets_file, "r", encoding="utf-8") as f:
                tickets = json.load(f)
            if not isinstance(tickets, list):
                tickets = []
        except (json.JSONDecodeError, IOError):
            tickets = []
    
    # Generate new ticket_id: TK-YYYYMMDD-NNN
    # Đếm số ticket trong ngày hôm nay để tạo sequence
    today_prefix = f"TK-{today}-"
    today_count = sum(
        1 for t in tickets
        if isinstance(t, dict) and str(t.get("ticket_id", "")).startswith(today_prefix)
    )
    seq = today_count + 1
    ticket_id = f"TK-{today}-{seq:03d}"
    
    # Build new ticket
    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat(),
    }
    
    # Append và lưu file
    tickets.append(new_ticket)
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    with open(tickets_file, "w", encoding="utf-8") as f:
        json.dump(tickets, f, ensure_ascii=False, indent=2)
    
    return {
        "ticket_id": ticket_id,
        "status": "open",
        "priority": priority,
        "customer_name": customer_name,
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# TODO: Định nghĩa JSON Schema cho từng tool (name, description, parameters).
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    # TODO: Thêm schema cho "search_product_catalog"
    {
        "type": "function",
        "function": {
            "name": "search_product_catalog",
            "description": (
                "Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa. "
                "Dùng khi khách hàng muốn tìm xe điện hoặc tour du lịch trong ngân sách."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["xe_dien", "du_lich"],
                        "description": "Loại sản phẩm: 'xe_dien' (xe điện) hoặc 'du_lich' (du lịch)."
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "Giá tối đa tính bằng VNĐ. Mặc định 999999999999 (không giới hạn)."
                    }
                },
                "required": ["category"]
            }
        }
    },
    # TODO: Thêm schema cho "submit_support_ticket"
    {
        "type": "function",
        "function": {
            "name": "submit_support_ticket",
            "description": (
                "Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket. "
                "Dùng khi khách hàng gặp sự cố hoặc cần hỗ trợ không thể giải quyết ngay."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "Tên khách hàng."
                    },
                    "issue_description": {
                        "type": "string",
                        "description": "Mô tả chi tiết vấn đề cần hỗ trợ."
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Mức độ ưu tiên. Mặc định 'medium'."
                    }
                },
                "required": ["customer_name", "issue_description"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}