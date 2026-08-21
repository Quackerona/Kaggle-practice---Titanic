# Phân tích dữ liệu khám phá (EDA)
*Đề tài: Kaggle: Titanic - Machine Learning from Disaster*

**Dữ liệu:** danh sách hành khách Titanic (`train.csv` / `test.csv`)
**Biến mục tiêu:** `Survived` (0 = không sống sót, 1 = sống sót)
 
## Tóm tắt tổng quan
 
Báo cáo này tìm hiểu các yếu tố ảnh hưởng đến khả năng sống sót của hành khách trên tàu Titanic, dựa trên 891 bản ghi trong tập train. Các phát hiện dưới đây được dùng làm cơ sở để định hướng xử lý dữ liệu và lựa chọn mô hình ở bước tiếp theo.
 
### Các phát hiện chính:

- Giới tính thể hiện rõ nhất sự sống sót (~74% phụ nữ sống sót so với ~18% của nam.)
- Càng nhiều thành viên gia đình => Càng ít cơ hội sống sót.  
- Những người trẻ sống sót nhiều hơn người già.
- Danh cấp cũng là một dấu hiệu lớn thể hiện sự sống sót.
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
 

## 2. Sử lý dữ liệu thiếu
train.csv:
| Cột | Số thiếu | Cách giải quyết |
| :--- | :--- | :--- |
| `Age` | 177 | Lấy trung vị của `Age`, tách bằng danh hiệu tên để chính xác hơn. |
| `Cabin` | 687 | Không sử dụng dữ liệu nên không cần làm gì. |
| `Embarked` | 2 | Không sử dụng dữ liệu nên không cần làm gì. |

test.csv:
| Cột | Số thiếu | Cách giải quyết |
| :--- | :--- | :--- |
| `Age` | 86 | Lấy trung vị của tuổi từ dữ liệu của train. |
| `Cabin` | 327 | Không sử dụng dữ liệu nên không cần làm gì. |
| `Fare` | 1 | Không sử dụng dữ liệu nên không cần làm gì. |

### Vì sao một số dữ liệu không sử dụng?
Các dữ liệu như `Cabin`, `Fare`, `Embarked` đều là về danh cấp. Chỉ cần sử dụng đến `Pclass` vì
1. `Pclass` là định danh trực tiếp.
2. Cả 3 cột này đều được gắn liền với `Pclass`, nếu sử dụng cả 3 sẽ dễ bị nhiễu thông tin.

## 3. Đề xuất sửa dữ liệu
Để dữ liệu đầu vào sạch và dễ hiểu cho model, có một số kỹ thuật có thể thực hiện:
- **Tách danh hiệu tên thành cột `Title` riêng** vì danh hiệu có thể là dấu hiệu của độ tuổi và danh cấp.

## 4. Triển khai model
Vì dữ liệu đơn giản và không liên tục, nên **Tree-based models** (RandomForest, XGBoost, ...) sẽ là hợp lý nhất cho chủ đề này.
- sử dụng **One-hot Encoding** cho các cột `Age`, `Sex`, `SibSp`, `Parch`
- cột `Pclass` cho thẳng vào ma trận, không cần encode vì đã mang tính chất ordinal.