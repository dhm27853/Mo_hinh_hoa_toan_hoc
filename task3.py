import time
import sys
from dd.autoref import BDD

# --- IMPORT TỪ FILE CŨ CỦA BẠN ---
try:
    # Import class PetriNet và Parser từ file task1.py
    from task1 import PNMLParser, PetriNet
except ImportError:
    print("❌ Lỗi: Không tìm thấy file 'task1.py'. Hãy chắc chắn file parser tên là task1.py")
    sys.exit(1)

def symbolic_reachability_bdd(pn: PetriNet):
    """
    Task 3: Tính toán tập trạng thái khả đạt (Reachability Set) bằng BDD.
    Sử dụng thư viện 'dd' của Python.
    """
    bdd = BDD()
    
    # ---------------------------------------------------------
    # BƯỚC 1: Khai báo biến BDD (Variables)
    # ---------------------------------------------------------
    # Với mỗi Place, ta cần 2 biến:
    # - var_curr (x): Trạng thái hiện tại
    # - var_next (x'): Trạng thái tiếp theo
    
    # Sắp xếp ID để đảm bảo thứ tự biến nhất quán
    place_ids = sorted(list(pn.places.keys()))
    
    # Tạo từ điển để tra cứu tên biến
    curr_vars = {p_id: f"x_{p_id}" for p_id in place_ids}        # Biến hiện tại
    next_vars = {p_id: f"x_{p_id}_prime" for p_id in place_ids}  # Biến tương lai
    
    # Khai báo với BDD manager
    bdd.declare(*curr_vars.values())
    bdd.declare(*next_vars.values())
    
    print(f"-> Đã khai báo {len(place_ids)*2} biến BDD.")

    # ---------------------------------------------------------
    # BƯỚC 2: Mã hóa Trạng thái Ban đầu (Initial Marking M0)
    # ---------------------------------------------------------
    # M0 = (x_p1 == init_val) & (x_p2 == init_val) ...
    
    initial_marking = pn.get_initial_marking()
    m0_parts = []
    
    for p_id in place_ids:
        # Nếu place có token -> biến đó là True (x_p), ngược lại là False (~x_p)
        if initial_marking.get(p_id, 0) > 0:
            m0_parts.append(f"{curr_vars[p_id]}")
        else:
            m0_parts.append(f"~ {curr_vars[p_id]}")
            
    m0_expr = " & ".join(m0_parts)
    # R là tập hợp các trạng thái đã tìm thấy. Ban đầu R = {M0}
    R = bdd.add_expr(m0_expr) 
    
    # ---------------------------------------------------------
    # BƯỚC 3: Xây dựng Hàm Chuyển đổi (Transition Relation - T)
    # ---------------------------------------------------------
    # T(x, x') = OR ( Enabled(t) & Change(t) & Frame(t) ) cho mọi t
    
    trans_bdd_list = []
    
    for t_id in pn.transitions:
        # Lấy danh sách Place đầu vào và đầu ra
        inputs = [p for p, w in pn.input_places.get(t_id, [])]
        outputs = [p for p, w in pn.output_places.get(t_id, [])]
        
        # a. Điều kiện kích hoạt (Enabled): Tất cả input phải = 1
        if not inputs: continue 
        enabled_expr = " & ".join([curr_vars[p] for p in inputs])
        
        # b. Cập nhật trạng thái (Update Logic & Frame Axiom)
        update_parts = []
        
        # Xác định các place bị ảnh hưởng bởi transition này
        affected = set(inputs).union(set(outputs))
        
        for p_id in place_ids:
            c_v = curr_vars[p_id]      # Biến hiện tại (x)
            n_v = next_vars[p_id]      # Biến tiếp theo (x')
            
            if p_id in inputs and p_id not in outputs:
                # Mất token: x -> 0 (x' = False)
                update_parts.append(f"~ {n_v}")
            elif p_id in outputs:
                # Nhận token: -> 1 (x' = True)
                update_parts.append(f"{n_v}")
            else:
                # Frame Axiom: Không liên quan thì giữ nguyên (x' <-> x)
                # Quan trọng: Nếu thiếu cái này, BDD sẽ sai!
                update_parts.append(f"({n_v} <-> {c_v})")
        
        update_expr = " & ".join(update_parts)
        
        # Kết hợp thành biểu thức cho 1 transition
        t_expr = f"({enabled_expr}) & ({update_expr})"
        trans_bdd_list.append(bdd.add_expr(t_expr))

    # Gộp tất cả transition lại: T = T1 | T2 | T3 ...
    T = bdd.false
    for t_bdd in trans_bdd_list:
        T = T | t_bdd
        
    print(f"-> Đã mã hóa {len(trans_bdd_list)} transitions.")

    # ---------------------------------------------------------
    # BƯỚC 4: Vòng lặp Tìm kiếm (Symbolic BFS)
    # ---------------------------------------------------------
    # R_new = R_curr OR Image(R_curr)
    
    print("-> Bắt đầu vòng lặp BDD...")
    start_time = time.time()
    steps = 0
    
    while True:
        steps += 1
        # Tính Image: Tìm tất cả trạng thái tiếp theo từ R hiện tại
        # Relprod = Existential Quantification AND Product
        # Ý nghĩa: Tồn tại trạng thái hiện tại sao cho (R đúng VÀ Transition đúng) -> trả về trạng thái tiếp theo
        # 1. Tính phép VÀ (Intersection/Conjunction) giữa Trạng thái hiện tại và Luật chuyển đổi
        intermediate = R & T

        # 2. Tính phép Tồn tại (Existential Quantification) để loại bỏ biến hiện tại (x)
        # Công thức: tồn tại x sao cho (R và T) đúng
        # Lưu ý: 'set(curr_vars.values())' là tập hợp tên các biến x_p1, x_p2...
        next_states_prime = bdd.exist(set(curr_vars.values()), intermediate)
        
        # Đổi tên biến: Biến tương lai (x') thành biến hiện tại (x) để gộp vào R
        rename_map = {next_vars[p]: curr_vars[p] for p in place_ids}
        next_states = bdd.let(rename_map, next_states_prime)
        
        # Hợp nhất với tập đã biết
        R_new = R | next_states
        
        # Kiểm tra điểm dừng (Fixpoint): Nếu không có trạng thái mới
        if R_new == R:
            break
            
        R = R_new
        # print(f"   Bước {steps}: Kích thước BDD = {len(R)}")

    end_time = time.time()
    
    # Đếm số lượng trạng thái thỏa mãn (SAT counting)
    count = bdd.count(R, nvars=len(place_ids))
    
    return int(count), end_time - start_time, steps

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Thay tên file PNML của bạn ở đây
    filename = "pnml_reachability.pnml"
    
    try:
        print(f"--- TASK 3: BDD REACHABILITY ---")
        
        # 1. Parse file dùng Task 1
        parser = PNMLParser()
        pn = parser.parse(filename)
        print(f"Đã đọc mô hình: {len(pn.places)} places, {len(pn.transitions)} transitions")
        
        # 2. Chạy thuật toán Task 3
        count, duration, steps = symbolic_reachability_bdd(pn)
        
        print("\n" + "="*40)
        print("KẾT QUẢ TASK 3")
        print("="*40)
        print(f"Tổng số Marking: {count}")
        print(f"Số bước lặp:     {steps}")
        print(f"Thời gian chạy:  {duration:.6f}s")
        print("="*40)
        
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{filename}'")
    except Exception as e:
        print(f"❌ Lỗi: {e}")