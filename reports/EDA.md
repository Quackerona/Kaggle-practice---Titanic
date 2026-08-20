# Phân tích dữ liệu khám phá (EDA)
*Đề tài: Kaggle: Titanic - Machine Learning from Disaster*

**Dữ liệu:** danh sách hành khách Titanic (`train.csv` / `test.csv`)
**Biến mục tiêu:** `Survived` (0 = không sống sót, 1 = sống sót)
 
## Tóm tắt tổng quan
 
Báo cáo này tìm hiểu các yếu tố ảnh hưởng đến khả năng sống sót của hành khách trên tàu Titanic, dựa trên 891 bản ghi trong tập train. Các phát hiện dưới đây được dùng làm cơ sở để định hướng xử lý dữ liệu và lựa chọn mô hình ở bước tiếp theo.
 
### Các phát hiện chính:

todo

## 1. Sơ lược về dữ liệu
 
Tập train gồm **891 dòng** và **12 cột**: 1 biến mục tiêu (`Survived`), 10 biến đặc trưng, và 1 mã định danh hành khách.
 
### Giải thích từng cột
 
| Cột | Kiểu | Ý nghĩa | Giá trị ví dụ |
| :--- | :--- | :--- | :--- |
| `PassengerId` | Số nguyên | Mã định danh riêng cho từng hành khách | 1 đến 891 |
| `Survived` | Số nguyên | Trạng thái sống sót | 0 = không, 1 = có |
| `Pclass` | Số nguyên | Hạng vé, cũng phản ánh mức kinh tế | 1, 2, hoặc 3 |
| `Name` | Chữ | Tên đầy đủ | "Braund, Mr. Owen Harris" |
| `Sex` | Chữ | Giới tính | `male`, `female` |
| `Age` | Số thập phân | Tuổi | 0.42 đến 80 |
| `SibSp` | Số nguyên | Số anh/chị/em hoặc vợ/chồng đi cùng | 0 đến 8 |
| `Parch` | Số nguyên | Số cha/mẹ hoặc con cái đi cùng | 0 đến 6 |
| `Ticket` | Chữ | Số vé | "A/5 21171" |
| `Fare` | Số thập phân | Giá vé (Bảng Anh) | £0 đến £512.33 |
| `Cabin` | Chữ | Số phòng | "C85", hoặc để trống |
| `Embarked` | Chữ | Cảng lên tàu | C, Q, hoặc S |
 

