# dacon study 2 - Book Recommend

 * 참여일: 23.04.24 ~ 23.05.15
 * 목적: 추천 "시스템" 만들어보자
 * 주제: 제2회 코스포 x 데이콘 도서 추천 알고리즘 AI경진대회(채용 연계형)
 * 경진대회 링크: [dacon link](https://dacon.io/competitions/official/236093)
 * 평가 방법: RMSE

 # ideation
 
 * 내 목적은 추천이었으나, 결국 이 경진대회의 목적은 시스템이 아닌 성능 좋은 예측 모형을 만들어 RMSE로 평가하는 것. 접근 방식을 바꿔야 함. 따라서 있는 정보를 모두 활용해서 최대한 높은 성능을 내는 것이 중요함. 이를 학습할 수 있도록 모델의 규모도 커지는 것이 성능을 높이는 데 좋을 것.
 

# 데이터 정보

* 데이터 위치: `open/`
*  Dataset Info. (아래와 같음)

**train.csv**

- ID : 샘플 고유 ID
- User-ID : 유저 고유 ID
- Book-ID : 도서 고유 ID

유저 정보
- Age : 나이
- Location : 지역

도서 정보
- Book-Title : 도서 명
- Book-Author : 도서 저자
- Year-Of-Publication : 도서 출판 년도 (-1일 경우 결측 혹은 알 수 없음)
- Publisher : 출판사
- Book-Rating : 유저가 도서에 부여한 평점 (0점 ~ 10점)

단, 0점인 경우에는 유저가 해당 도서에 관심이 없고 관련이 없는 경우

**test.csv**
- ID : 샘플 고유 ID
- User-ID : 유저 고유 ID
- Book-ID : 도서 고유 ID

유저 정보
- Age : 나이
- Location : 지역

도서 정보
- Book-Title : 도서 명
- Book-Author : 도서 저자
- Year-Of-Publication : 도서 출판 년도 (-1일 경우 결측 혹은 알 수 없음)
- Publisher : 출판사

**sample_submission.csv**
- ID : 샘플 고유 ID
- Book-Rating : 예측한 유저가 도서에 부여할 평점

단, 0점인 경우에는 유저가 해당 도서에 관심이 없고 관련이 없는 경우

※ 본 경진대회는 추천시스템 주제에서 널리 알려져 있는 Book-Crossing 오픈 데이터셋을 기반으로 진행됩니다.