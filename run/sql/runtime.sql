CREATE TABLE runtime
(
    id           bigint        NOT NULL AUTO_INCREMENT COMMENT '主键',
    run_id       varchar(64)   NOT NULL DEFAULT '' COMMENT '运行id',
    batch_run_id varchar(64)   NOT NULL DEFAULT '' COMMENT '批量运行id',
    config       varchar(2045) NOT NULL DEFAULT '' COMMENT '配置',
    create_time  timestamp     NOT NULL DEFAULT current_timestamp COMMENT '创建时间',
    code         int           NOT NULL DEFAULT 0  COMMENT '失败原因，1表示成功',
    tag          varchar(10)   NOT NULL DEFAULT '' COMMENT '标记',
    seed         int(10)       NOT NULL DEFAULT 0 COMMENT '随机种子',
    PRIMARY KEY (id)
) ENGINE=innoDB DEFAULT CHARSET=utf8 comment '运行时配置';
