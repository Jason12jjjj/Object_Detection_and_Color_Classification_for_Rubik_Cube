# 請將這段貼進你的 rubiks_core.py 替換掉原本的 solve_cube

import kociemba

def solve_cube(faces_data):
    """Run Kociemba with EXHAUSTIVE error checking and fool-proofing."""
    try:
        # 1. 確保字串轉換正確 (這裡假設你的 to_kociemba_string 已經寫好)
        cube_string = to_kociemba_string(faces_data)
        
        # 2. 呼叫 Kociemba 求解
        solution = kociemba.solve(cube_string)
        return True, solution

    except ValueError as e:
        # Kociemba 拋出的 ValueError 包含特定的錯誤描述，我們必須攔截並翻譯
        error_msg = str(e).lower()
        
        if "not exactly one facelet of each colour" in error_msg:
            return False, "❌ Error: Invalid color count. Please check the sticker stats."
        elif "not all 12 edges exist exactly once" in error_msg:
            return False, "❌ Physics Error: Invalid edge pieces detected. (e.g., A piece with two identical colors)."
        elif "one edge has to be flipped" in error_msg:
            return False, "❌ Parity Error: An edge piece is flipped in an impossible way. Please re-scan."
        elif "not all 8 corners exist exactly once" in error_msg:
            return False, "❌ Physics Error: Invalid corner pieces detected. You might have painted the wrong colors on a corner."
        elif "one corner has to be twisted" in error_msg:
            return False, "❌ Parity Error: A corner is physically twisted. This state cannot exist on a real Rubik's cube."
        elif "two corners or two edges have to be exchanged" in error_msg:
            return False, "❌ Parity Error: Two pieces are swapped. This requires breaking the cube to fix. Check your colors."
        else:
            return False, f"❌ Invalid Cube State: {str(e)}"
            
    except Exception as exc:
        # 系統級別的崩潰錯誤攔截
        return False, f"🚨 Critical Solver Error: {exc}"
