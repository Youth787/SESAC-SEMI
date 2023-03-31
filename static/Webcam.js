document.addEventListener('DOMContentLoaded', function() {
  const videoElement = document.getElementById('video-element');
  
  if (videoElement) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        videoElement.srcObject = stream;
        videoElement.play();
      })
  }
  });


  function takeSnapshot() {
    const videoElement = document.getElementById('video-element');
    var myCanvasElement = document.getElementById('myCanvas');
    const personshape = document.getElementById('person-shape');
    var myCTX = myCanvasElement.getContext('2d');
    const form = document.getElementById('simul');
    var videoAspectRatio = videoElement.videoWidth / videoElement.videoHeight;
    var canvasAspectRatio = myCanvasElement.width / myCanvasElement.height;
  
    var cropWidth, cropHeight, xOffset, yOffset;
    if (videoAspectRatio > canvasAspectRatio) {
      cropWidth = videoElement.videoHeight * canvasAspectRatio;
      cropHeight = videoElement.videoHeight;
      xOffset = (videoElement.videoWidth - cropWidth) /2;
      yOffset = 0;
    } else {
      cropWidth = videoElement.videoWidth;
      cropHeight = videoElement.videoWidth / canvasAspectRatio;
      xOffset = 0;
      yOffset = (videoElement.videoHeight - cropHeight) /2;
    }
  
      myCTX.drawImage(videoElement, xOffset, yOffset, cropWidth, cropHeight, 0, 0, 300, 200);
      saveSnapshot(myCanvasElement.toDataURL());
      personshape.style.display = "none";
      videoElement.style.display = "none";
      myCanvasElement.style.display = "none";
      form.style.display = "none";
  }
  

  function saveSnapshot(dataURL) {
    
    $.ajax({
      type: "POST",
      url: "save-snapshot/",
      data: { image_data: dataURL },
      success: function() {
        console.log("Snapshot saved successfully!");
        $('#simul').submit();
      },
      error: function() {
        console.log("Failed to save snapshot");
      }
    });
  }
  
  


  IMP.request_pay({
    pg : 'inicis',
    pay_method : 'card',
    merchant_uid : 'merchant_' + new Date().getTime(),
    name : '주문명:결제테스트',
    amount : 14000,
    buyer_email : 'iamport@siot.do',
    buyer_name : '구매자이름',
    buyer_tel : '010-1234-5678',
    buyer_addr : '서울특별시 강남구 삼성동',
    buyer_postcode : '123-456'
}, function(rsp) {
    if ( rsp.success ) {
     //[1] 서버단에서 결제정보 조회를 위해 jQuery ajax로 imp_uid 전달하기
     jQuery.ajax({
      url: "/payments/complete", //cross-domain error가 발생하지 않도록 동일한 도메인으로 전송
      type: 'POST',
      dataType: 'json',
      data: {
       imp_uid : rsp.imp_uid
       //기타 필요한 데이터가 있으면 추가 전달
      }
     }).done(function(data) {
      //[2] 서버에서 REST API로 결제정보확인 및 서비스루틴이 정상적인 경우
      if ( everythings_fine ) {
       var msg = '결제가 완료되었습니다.';
       msg += '\n고유ID : ' + rsp.imp_uid;
       msg += '\n상점 거래ID : ' + rsp.merchant_uid;
       msg += '\결제 금액 : ' + rsp.paid_amount;
       msg += '카드 승인번호 : ' + rsp.apply_num;

       alert(msg);
      } else {
       //[3] 아직 제대로 결제가 되지 않았습니다.
       //[4] 결제된 금액이 요청한 금액과 달라 결제를 자동취소처리하였습니다.
      }
     });
    } else {
        var msg = '결제에 실패하였습니다.';
        msg += '에러내용 : ' + rsp.error_msg;

        alert(msg);
    }
});



var IMP = window.IMP; 
IMP.init("imp48532846"); 

function requestPay() {
    IMP.request_pay({
        pg : 'kcp.{상점ID}',
        pay_method : 'card',
        merchant_uid: "57008833-33005", 
        name : '전동킥보드',
        amount : 100,
        buyer_email : 'Iamport@chai.finance',
        buyer_name : '포트원 기술지원팀',
        buyer_tel : '010-1234-5678',
        buyer_addr : '서울특별시 강남구 삼성동',
        buyer_postcode : '123-456'
    }, function (rsp) { // callback
        if (rsp.success) {
            window.location.href = "http://127.0.0.1:8000/yolo/";
        } else {
            console.log("이미 결재.");
            window.location.href = "http://127.0.0.1:8000/yolo/";
        }
    });
}