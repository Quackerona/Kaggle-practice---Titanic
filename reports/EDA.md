# Phân tích dữ liệu khám phá (EDA)

**Đề tài:** Kaggle: Titanic - Machine Learning from Disaster
**Dữ liệu:** danh sách hành khách Titanic (`train.csv` / `test.csv`)
**Biến mục tiêu:** `Survived` (0 = không sống sót, 1 = sống sót)

## Tóm tắt tổng quan

Báo cáo này tìm hiểu các yếu tố ảnh hưởng đến khả năng sống sót của hành khách trên tàu Titanic, dựa trên 891 bản ghi trong tập train. Các phát hiện dưới đây được dùng làm cơ sở để định hướng xử lý dữ liệu và lựa chọn mô hình ở bước tiếp theo.

### Các phát hiện chính:

- Giới tính thể hiện rõ nhất sự sống sót (~74% phụ nữ sống sót so với ~18% của nam.)
- Càng nhiều thành viên gia đình => Càng ít cơ hội sống sót.
- Những người trẻ sống sót nhiều hơn người già, nhưng chênh lệch không nhiều bằng mấy cái trên.
- Danh cấp cũng là một dấu hiệu lớn thể hiện sự sống sót.
- Danh hiệu tên (`Title`) là cái mạnh nhất trong số các biến phân loại được test, mạnh hơn cả `Sex` và `Pclass`.

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
| `Cabin` | 687 | Tách chữ cái đầu thành `Deck`, dòng nào thiếu thì cho là "U" (Unknown). |
| `Embarked` | 2 | Điền bằng giá trị xuất hiện nhiều nhất. |

test.csv:
| Cột | Số thiếu | Cách giải quyết |
| :--- | :--- | :--- |
| `Age` | 86 | Lấy trung vị của tuổi từ dữ liệu của train. |
| `Cabin` | 327 | Làm giống như trên, tách ra `Deck`. |
| `Fare` | 1 | Điền bằng trung vị `Fare` theo `Pclass` tương ứng. |

## 3. Test xem biến nào thực sự quan trọng


**Biến phân loại (chi-square), xếp từ mạnh xuống yếu:**

| Biến | Chi2 | p-value |
| :--- | :--- | :--- |
| `Title` | 300.02 | < 0.000001 |
| `Sex` | 260.72 | < 0.000001 |
| `Pclass` | 102.89 | < 0.000001 |
| `Deck` | 99.16 | < 0.000001 |
| `FamilySize` | 80.67 | < 0.000001 |
| `SibSp` | 37.27 | 0.000002 |
| `Parch` | 27.93 | 0.000097 |
| `Embarked` | 26.49 | 0.000002 |

**Biến số (t-test):**

| Biến | Mean (sống sót) | Mean (không sống sót) | p-value |
| :--- | :--- | :--- | :--- |
| `Fare` | 48.40 | 22.12 | < 0.000001 |
| `Age` | 28.34 | 30.63 | 0.041 |

Tất cả đều có p-value < 0.05 nên đều quan trọng, không phải ngẫu nhiên. Vài điều cần lưu ý:

- `Age` có p-value = 0.041, gần sát ngưỡng 0.05 nhất trong hết. Nên đây là biến yếu nhất trong danh sách này.
- `Deck` có nhóm `T` chỉ 1 người thôi. Kết quả chung vẫn ổn nhờ mấy nhóm khác đủ lớn, nhưng riêng nhóm `T` thì không nên tin quá.
- Gộp `SibSp` với `Parch` thành `FamilySize` thấy hợp lý hơn, vì `FamilySize` có chi-square cao hơn 2 cái kia cộng lại.
- Mấy con số này chỉ nói lên là "biến đó có liên quan", chưa nói được là model sẽ dùng nó nhiều hay ít.Sau khi train model rồi xem `feature_importance` mới biết chắc.

## 4. Chọn feature nào để dùng

Có ý nghĩa thống kê không có nghĩa là nên nhét hết vào model vì sẽ trùng thông tin. Để biết 2 biến có trùng nhau không, dùng:

- **Cramér's V**: dùng khi cả 2 biến đều là loại categorical, ví dụ `Pclass` với `Deck`.
- **Correlation ratio**: dùng khi 1 biến là categorical, 1 biến là số, ví dụ `Pclass` với `Fare`.


Giá trị càng gần 1 là càng trùng nhau, càng gần 0 là càng độc lập.

**Nhóm liên quan tới tiền/hạng vé (`Pclass`, `Fare`, `Deck`):**

| Cặp biến | Giá trị |
| :--- | :--- |
| Cramér's V (`Pclass` vs `Deck`) | 0.605 |
| Correlation ratio (`Pclass` vs `Fare`) | 0.594 |
| Correlation ratio (`Deck` vs `Fare`) | 0.577 |

Cả 3 chỉ ở mức trung bình (~0.58-0.60), không quá cao, nên 3 biến này vẫn có thông tin riêng, không hoàn toàn trùng nhau. Giữ cả 3 làm ứng viên được.

**Nhóm liên quan tới giới tính/tuổi (`Sex`, `Age`, `Title`):**

| Cặp biến | Giá trị |
| :--- | :--- |
| Cramér's V (`Sex` vs `Title`) | 0.998 |
| Correlation ratio (`Title` vs `Age`) | 0.573 |
| Correlation ratio (`Sex` vs `Age`) | 0.105 |

`Sex` và `Title` gần như trùng hoàn toàn (0.998) vì `Title` (Mr/Mrs/Miss/Master) gần như nói lên luôn giới tính rồi. Cho cả 2 vào model là dư. Vì `Title` còn có thêm thông tin (ví dụ "Master" là bé trai, tách biệt với đàn ông trưởng thành, còn `Sex` thì không phân biệt được), nên giữ `Title`, bỏ `Sex`.

### Tóm lại nên chọn feature nào

- Nhóm gia đình: dùng `FamilySize`, bỏ `SibSp`/`Parch` riêng lẻ.
- Nhóm giới tính/tuổi: dùng `Title` (đã có gần hết thông tin của `Sex` rồi), giữ `Age` riêng vì không trùng nhiều với `Title`.
- Nhóm tiền/hạng vé: giữ cả `Pclass`, `Fare`, `Deck` vì không trùng nhau nhiều.
- Cuối cùng vẫn nên đưa hết mấy feature còn lại (`Title`, `Age`, `FamilySize`, `Pclass`, `Fare`, `Deck`, `Embarked`) vào train thử, rồi xem `feature_importance` để cắt bớt cái nào đóng góp ít.
## 5. Đề xuất sửa dữ liệu

Để dữ liệu đầu vào sạch và dễ hiểu cho model, có một số kỹ thuật có thể thực hiện:

- **Tách danh hiệu tên thành cột `Title` riêng** vì danh hiệu có thể là dấu hiệu của độ tuổi và danh cấp.
- **Tạo cột `Deck`** bằng cách lấy chữ cái đầu của `Cabin`, dòng thiếu thì cho là "U".
- **Tạo cột `FamilySize` = `SibSp` + `Parch` + 1** để gộp lại thành 1 biến duy nhất, dễ nhìn hơn để riêng.

## 6. Kết luận rút ra từ EDA

Từ mấy cái ở trên, rút ra được vài điều cho bước xử lý dữ liệu và chọn model:

- **Giới tính và danh hiệu tên là 2 yếu tố mạnh nhất**: `Title` (chi2 = 300.02) và `Sex` (chi2 = 260.72) đứng đầu, cao hơn hẳn mấy cái còn lại. `Title` gộp luôn giới tính, tuổi, và đôi khi cả địa vị xã hội (`Master`, `Lady`, `Sir`,...) vào 1 biến, nên tách `Title` ra là bước quan trọng.
- **Gộp `SibSp` với `Parch` thành `FamilySize` là đúng**: `FamilySize` (chi2 = 80.67) mạnh hơn cả 2 cái gốc, nên gộp lại hợp lý hơn để riêng.
- **Mấy biến còn lại đều có ý nghĩa nhưng không đều nhau**: `Pclass`, `Deck`, `Embarked`, `Fare` đều p < 0.05, riêng `Age` thì p = 0.041, gần sát ngưỡng nên là biến yếu nhất.
- **p-value chỉ nói được là biến có liên quan thôi**: chưa nói được nó sẽ đóng góp nhiều hay ít khi train chung với mấy biến khác. Bước sau nên thử train rồi xem `feature_importance` để biết chắc.
- **`Sex` và `Title` trùng nhau gần hết**: Cramér's V = 0.998, nên chỉ cần giữ `Title` thôi (xem mục 4).

## 7. Triển khai model

Vì dữ liệu đơn giản và không liên tục, nên **Tree-based models** (RandomForest, XGBoost, ...) sẽ là hợp lý nhất cho chủ đề này.

- sử dụng **One-hot Encoding** cho các cột `Embarked`, `Title`, `Deck`
- cột `Pclass` cho thẳng vào ma trận, không cần encode vì đã mang tính chất ordinal.
- không cho `Sex` vào chung với `Title` để tránh dư thừa (xem mục 4).
- sau khi train xong, xem `feature_importance` để check lại xem `Fare`, `Deck`, `Embarked` có thực sự đóng góp không, thay vì tự quyết định trước.