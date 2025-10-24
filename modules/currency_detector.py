import cv2
import numpy as np

class INRCurrencyDetector:
    def __init__(self):
        self.min_area_ratio = 0.02
        self.max_area_ratio = 0.6

    def detect_currency(self, frame):
        h, w = frame.shape[:2]
        frame_area = h * w

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < 1000: continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            rect_area = bw * bh
            area_ratio = rect_area / frame_area
            if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
                continue
            aspect = bw / (bh + 1e-6)
            aspect_score = max(0, 1 - abs(aspect - 2.15) / 0.6)
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            quad_score = 1.0 if len(approx) == 4 else 0.6 if 4 < len(approx) < 8 else 0.2
            score = 0.6 * aspect_score + 0.4 * quad_score
            if score > best_score:
                best_score = score; best = (x, y, bw, bh, aspect)

        res = {
            "currency_detected": False, "denomination": "unknown",
            "confidence": float(best_score), "position": None,
            "orientation": "unknown", "condition": "unknown"
        }
        if best and best_score > 0.45:
            x,y,bw,bh,aspect = best
            res["currency_detected"] = True
            res["position"] = (x,y,bw,bh)
            res["orientation"] = "horizontal" if aspect >= 1 else "vertical"
            roi = frame[y:y+bh, x:x+bw]
            res["denomination"] = self._guess_inr_denomination(roi)

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges_roi = cv2.Canny(gray_roi, 50, 150)
            edge_density = np.mean(edges_roi > 0)
            std = float(gray_roi.std())
            if edge_density > 0.12 or std > 55: res["condition"] = "worn or damaged"
            elif std < 18: res["condition"] = "good condition"
            else: res["condition"] = "fair condition"
        return res

    def _guess_inr_denomination(self, roi):
        if roi.size == 0: return "unknown"
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        def pct(l,u):
            mask = cv2.inRange(hsv, np.array(l), np.array(u))
            return 100.0 * (cv2.countNonZero(mask) / (roi.shape[0]*roi.shape[1]))
        purple = pct((130,40,40),(165,255,255))
        yellow = pct((20,40,40),(35,255,255))
        green  = pct((40,40,40),(80,255,255))
        pink   = pct((165,30,40),(179,255,255)) + pct((0,30,40),(10,255,255))
        orange = pct((10,40,40),(20,255,255))
        cands = [("₹100", purple), ("₹200", yellow), ("₹500", green), ("₹2000", pink), ("₹10/₹20", orange)]
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands[0][0] if cands[0][1] > 6 else "unknown"

    def guidance_text(self, results):
        if not results.get("currency_detected"):
            return "No currency detected. Place the note flat, in good light, and fill more of the camera view."
        parts = ["Indian currency detected."]
        denom = results.get("denomination","unknown")
        parts.append(f"Denomination looks like {denom}." if denom!="unknown" else "Denomination unclear.")
        ori = results.get("orientation","unknown")
        parts.append("The note is horizontal." if ori=="horizontal" else "The note is vertical." if ori=="vertical" else "The note seems tilted.")
        cond = results.get("condition","unknown")
        if cond == "worn or damaged": parts.append("The note appears worn or damaged.")
        elif cond == "good condition": parts.append("The note appears in good condition.")
        pos = results.get("position")
        if pos:
            x,y,w,h = pos
            frame_center_x = 320
            note_center_x = x + w//2
            if note_center_x < frame_center_x - 50: parts.append("Move the note slightly to the right.")
            elif note_center_x > frame_center_x + 50: parts.append("Move the note slightly to the left.")
            else: parts.append("The note is centered.")
        return " ".join(parts)

_detector = INRCurrencyDetector()
def detect_currency_in_frame(frame): return _detector.detect_currency(frame)
def get_currency_guidance_text(results): return _detector.guidance_text(results)
